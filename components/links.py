import math
import abc
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, \
    TransformerDecoderLayer
import torch.nn.functional as F


from transformers import BartModel, AutoModel
from transformers.models.bart.modeling_bart import BartAttention
EPSILON = 1e-8
BART_LINK = "facebook/bart-base"
BARTHEZ_LINK= "moussaKam/barthez"
LOCAL_ONLY = True

# ============================================== BASE CLASSES ==========================================================

class BaseLink(nn.Module):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, dropout=0.,
                 batchnorm=False, residual=None):
        super(BaseLink, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.z_size = z_size
        self.depth = depth
        self.params = params
        self.embedding = embedding
        self.highway = highway
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.prev_state = None
        self.next_state = None


class SequentialLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, dropout=0.,
                 batchnorm=False, residual=None):
        super(SequentialLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                             dropout=dropout, batchnorm=batchnorm, residual=residual)


class NamedLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, dropout=0.,
                 batchnorm=False, residual=None):
        super(NamedLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                             dropout=dropout, batchnorm=batchnorm, residual=residual)


# ============================================== LINK CLASSES ==========================================================


class MLPLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None):
        super(MLPLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout, batchnorm=batchnorm, residual=residual)

        if depth>1:
            self.mlp = nn.ModuleList([nn.Linear(input_size, output_size)] +
                                     [nn.Linear((input_size + output_size*i) if self.highway else output_size, output_size)
                                      for i in range(2, depth)])
        else:
            self.mlp = []
        self.drp_layer = torch.nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(z_size)

        mlp_out_size = ((output_size * depth) if self.highway else output_size) if depth > 1 else input_size
        if embedding is not None:
            assert mlp_out_size == embedding.weight.shape[1], "The MLP output size {} while the embedding size is " \
                                                              "{}".format(mlp_out_size, embedding.weight.shape[1])
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size)
                                                     for param in params})
            self.hidden_to_z_params['logits'].weight = embedding.weight
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            x_res = self.drp_layer(x_res)
            z_params_res = self.residual['link'](x_res, z_prev)
        x = self.drp_layer(x)
        outputs = []
        input = x
        for layer in self.mlp:
            outputs.append(layer(input))
            input = torch.cat([input, self.drp_layer(F.gelu(outputs[-1]))], dim=-1) if self.highway else outputs[-1]

        outputs = (torch.cat(outputs, dim=-1) if self.highway else outputs[-1]) if len(self.mlp) else input

        z_params = {param: activation(self.hidden_to_z_params[param](outputs)) for param, activation in
                    self.params.items()}

        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        #z_params = {'logits': self.hidden_to_logits(self.hidden_to_z_params['logits'](self.mlp[0](self.drp_layer(x))))}
        #z_params = {'logits': self.hidden_to_logits((self.drp_layer(x)))}
        return z_params


class DANLink(BaseLink):
    # Deep Averaging Network link
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None):
        super(DANLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout, batchnorm=batchnorm, residual=residual)

        if depth>1:
            self.mlp = nn.ModuleList([nn.Linear(input_size, output_size)] +
                                     [nn.Linear((input_size + output_size*i) if self.highway else output_size, output_size)
                                      for i in range(2, depth)])
        else:
            self.mlp = []
        self.drp_layer = torch.nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(z_size)

        mlp_out_size = ((output_size * depth) if self.highway else output_size) if depth > 1 else input_size
        if embedding is not None:
            assert mlp_out_size == embedding.weight.shape[1], "The MLP output size {} while the embedding size is " \
                                                              "{}".format(mlp_out_size, embedding.weight.shape[1])
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size)
                                                     for param in params})
            self.hidden_to_z_params['logits'].weight = embedding.weight
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            x_res = self.drp_layer(x_res)
            z_params_res = self.residual['link'](x_res, z_prev)
        x = self.drp_layer(x)
        outputs = []
        input = x.mean(-2)
        for layer in self.mlp:
            outputs.append(layer(input))
            input = torch.cat([input, self.drp_layer(F.gelu(outputs[-1]))], dim=-1) if self.highway else outputs[-1]

        outputs = (torch.cat(outputs, dim=-1) if self.highway else outputs[-1]) if len(self.mlp) else input
        outputs = outputs.unsqueeze(-2).repeat(*([1]*(outputs.ndim-1)), x.shape[-2], 1)

        z_params = {param: activation(self.hidden_to_z_params[param](outputs)) for param, activation in
                    self.params.items()}

        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        #z_params = {'logits': self.hidden_to_logits(self.hidden_to_z_params['logits'](self.mlp[0](self.drp_layer(x))))}
        #z_params = {'logits': self.hidden_to_logits((self.drp_layer(x)))}
        return z_params


class LastStateMLPLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None):
        super(LastStateMLPLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout, batchnorm=batchnorm, residual=residual)

        if depth>1:
            self.mlp = nn.ModuleList([nn.Linear(input_size, output_size)] +
                                     [nn.Linear((input_size + output_size*i) if self.highway else output_size, output_size)
                                      for i in range(2, depth)])
        else:
            self.mlp = []
        self.drp_layer = torch.nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(z_size)

        mlp_out_size = ((output_size * depth) if self.highway else output_size) if depth > 1 else input_size
        if embedding is not None:
            assert mlp_out_size == embedding.weight.shape[1], "Output size ({}) and embedding size ({}) are " \
                                                              "different".format(mlp_out_size,
                                                                                 embedding.weight.shape[1])
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size)
                                                     for param in params})
            self.hidden_to_z_params['logits'].weight = embedding.weight
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            x_res = self.drp_layer(x_res)
            z_params_res = self.residual['link'](x_res, z_prev)

        seq_size = x.shape[-2]
        x = x[..., -1, :]
        x = self.drp_layer(x)
        outputs = []
        input = x
        for layer in self.mlp:
            outputs.append(layer(input))
            input = torch.cat([input, self.drp_layer(F.gelu(outputs[-1]))], dim=-1) if self.highway else outputs[-1]

        outputs = (torch.cat(outputs, dim=-1) if self.highway else outputs[-1]) if len(self.mlp) else input

        z_params = {param: activation(self.hidden_to_z_params[param](outputs)) for param, activation in
                    self.params.items()}
        for k, v in z_params.items():
            z_params[k] = v.unsqueeze(-2).expand(*v.shape[:-1], seq_size, v.shape[-1])

        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        #z_params = {'logits': self.hidden_to_logits(self.hidden_to_z_params['logits'](self.mlp[0](self.drp_layer(x))))}
        #z_params = {'logits': self.hidden_to_logits((self.drp_layer(x)))}
        return z_params


class LSTMLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, last_state=False, bidirectional=False):
        super(LSTMLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout, batchnorm=batchnorm, residual=residual)

        self.rnn = nn.LSTM(input_size, output_size, depth, batch_first=False, bidirectional=bidirectional,
                           dropout=dropout)
        self.last_state = last_state
        self.bidirectional = bidirectional

        self.drp_layer = torch.nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(z_size)

        if bidirectional:
            output_size *= 2
        if embedding is not None:
            assert output_size == embedding.weight.shape[1], 'output size {} is different from ' \
                                                             'embedding size {}'.format(output_size,
                                                                                        embedding.weight.shape[1])
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size)
                                                     for param in params})
            self.hidden_to_z_params['logits'].weight = embedding.weight
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            x_res = self.drp_layer(x_res)
            z_params_res = self.residual['link'](x_res, z_prev)

        if x.ndim>3:
            batch_shape = x.shape[:-2]
            x = x.view(-1, *x.shape[-2:])
        else:
            batch_shape = None
        x = self.drp_layer(x)

        x = x.transpose(0, 1)
        if lens is None:
            device = next(self.parameters()).device
            lens = torch.ones(x.shape[1], device=device) * x.shape[0]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_x, self.prev_state)
        self.next_state = (hidden, cell)
        if self.last_state:
            outputs = torch.cat([hidden[-1, :, :], hidden[-2, :, :]] if self.bidirectional else
                                [hidden[-1, :, :]], dim=1)
            outputs = outputs.unsqueeze(1).repeat(1, x.shape[0], 1)
        else:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=x.shape[0])
        if batch_shape is not None:
            outputs = outputs.view(*batch_shape, *outputs.shape[-2:])

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}

        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        #z_params = {'logits': self.hidden_to_logits(self.hidden_to_z_params['logits'](self.mlp[0](self.drp_layer(x))))}
        #z_params = {'logits': self.hidden_to_logits((self.drp_layer(x)))}
        return z_params


class SublevelLSTMLink(NamedLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, last_state=False, bidirectional=False, sub_lvl_vars=None,
                 sub_lvl_size=None, up_lvl=False):
        assert (sub_lvl_size is not None) and (sub_lvl_vars is not None)
        super(SublevelLSTMLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout, batchnorm=batchnorm, residual=residual)
        self.sub_lvl_size, self.sub_lvl_vars = sub_lvl_size, sub_lvl_vars
        output_size = int(output_size/self.sub_lvl_size) if not (last_state or up_lvl) else output_size
        self.rnn = nn.LSTM(input_size, output_size, depth, batch_first=False, bidirectional=bidirectional,
                           dropout=dropout)

        self.up_rnn = nn.LSTM(output_size*2 if bidirectional else output_size, output_size, 1, batch_first=False,
                              bidirectional=bidirectional, dropout=dropout) if up_lvl else None
        self.last_state = last_state
        self.bidirectional = bidirectional

        self.drp_layer = torch.nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(z_size)

        if bidirectional:
            output_size *= 2
        if embedding is not None:
            assert output_size == embedding.weight.shape[1], 'output size {} is different from ' \
                                                             'embedding size {}'.format(output_size,
                                                                                        embedding.weight.shape[1])
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size)
                                                     for param in params})
            self.hidden_to_z_params['logits'].weight = embedding.weight
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            x_res = self.drp_layer(x_res)
            z_params_res = self.residual['link'](x_res, z_prev)
        sub_lvl_x = []
        for k, v in x.items():
            if k in self.sub_lvl_vars:
                assert v.shape[-1] % self.sub_lvl_size==0, "Please make the dimension of your sublevel input {}:{} " \
                                                           "divisible by the sequence length " \
                                                           "{}".format(k, v.shape[-1], self.sub_lvl_size)
                sub_lvl_x.append(v.reshape(*v.shape[:-1], self.sub_lvl_size,
                                              int(v.shape[-1]/self.sub_lvl_size)))
            else:
                expand_arg = (*v.shape[:-1], self.sub_lvl_size, v.shape[-1])
                sub_lvl_x.append(v.unsqueeze(-2).expand(expand_arg))

        x = torch.cat(sub_lvl_x, -1)
        if x.ndim>3:
            batch_shape = x.shape[:-2]
            x = x.view(-1, *x.shape[-2:])
        else:
            batch_shape = None
        x = self.drp_layer(x)

        x = x.transpose(0, 1)
        if lens is None:
            device = next(self.parameters()).device
            lens = torch.ones(x.shape[1], device=device) * x.shape[0]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False)

        packed_outputs, (hidden, cell) = self.rnn(packed_x)

        if self.last_state or self.up_rnn is not None:
            outputs = torch.cat([hidden[-1, :, :], hidden[-2, :, :]] if self.bidirectional else
                                [hidden[-1, :, :]], dim=1)
        else:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=x.shape[0])

        if batch_shape is not None:
            if self.last_state or self.up_rnn is not None:
                outputs = outputs.view(*batch_shape, outputs.shape[-1])
            else:
                outputs = outputs.view(*batch_shape, *outputs.shape[-2:])

        if self.up_rnn is not None:
            up_inputs = outputs
            if up_inputs.ndim > 3:
                batch_shape = up_inputs.shape[:-2]
                up_inputs = up_inputs.view(-1, *up_inputs.shape[-2:])
            else:
                batch_shape = None
            up_inputs = up_inputs.transpose(0, 1)
            device = next(self.parameters()).device
            lens = torch.ones(up_inputs.shape[1], device=device) * up_inputs.shape[0]
            packed_x = nn.utils.rnn.pack_padded_sequence(up_inputs, lens, enforce_sorted=False)
            packed_outputs, (hidden, cell) = self.up_rnn(packed_x)
            if self.last_state:
                outputs = torch.cat([hidden[-1, :, :], hidden[-2, :, :]] if self.bidirectional else
                                    [hidden[-1, :, :]], dim=1)
                outputs = outputs.unsqueeze(-2).repeat(1, up_inputs.shape[0], 1)
            else:
                outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                              total_length=up_inputs.shape[0])

            if batch_shape is not None:
                if self.last_state:
                    outputs = outputs.view(*batch_shape, outputs.shape[-1])
                else:
                    outputs = outputs.view(*batch_shape, *outputs.shape[-2:])

        z_params = {param: activation(self.hidden_to_z_params[param](outputs)) for param, activation in
                    self.params.items()}

        if not (self.last_state or (self.up_rnn is not None)):
            z_params = {k: v.view(*v.shape[:-2], v.shape[-1]*v.shape[-2]) for k, v in z_params.items()}

        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        return z_params


class GRULink(SequentialLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None):
        super(GRULink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout, batchnorm=batchnorm, residual=residual)
        self.rnn = nn.GRU(input_size, output_size, depth, batch_first=True, dropout=dropout)
        self.drp_layer = torch.nn.Dropout(p=dropout)
        rnn_output_size = (output_size*depth) if highway else output_size
        self.bn = nn.BatchNorm1d(z_size)
        if embedding is not None:
            self.sbn = sbn
            self.project_z_prev = nn.Linear(embedding.weight.shape[1], output_size * depth)
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(rnn_output_size, z_params_size)
                                                     for param in params})
        else:
            self.project_z_prev = nn.Linear(z_size, output_size*depth)
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(rnn_output_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            x_res = self.drp_layer(x_res)
            z_params_res = self.residual['link'](x_res, z_prev)
        x = self.drp_layer(x)
        h_prev = self.project_z_prev(z_prev) if z_prev is not None else None
        h_prev = h_prev.view(-1, self.depth, int(h_prev.shape[-1]/self.depth)).transpose(0, 1).contiguous()\
            if h_prev is not None else None
        flatten = x.ndim > 2
        if flatten:
            orig_shape = x.shape
            x = x.reshape(-1, orig_shape[-1])
        rnn_out, h = self.rnn(x.unsqueeze(-2), hx=h_prev)
        # These parameters are those of q(zi|xi, z<i) for the current word i (even though it's called z_params and
        # not zi_params)
        if self.highway:
            reshaped_h = h.transpose(0, 1).reshape(x.shape[0], self.output_size * self.depth)
        else:
            reshaped_h = rnn_out.squeeze(1)
        if flatten:
            reshaped_h = reshaped_h.view(*orig_shape[:-1], reshaped_h.shape[-1])

        reshaped_h = self.drp_layer(reshaped_h)
        z_params = {param: activation(self.hidden_to_z_params[param](reshaped_h))+EPSILON
                    for param, activation in self.params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        return z_params


class TransformerLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False):
        super(TransformerLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                              dropout=dropout, batchnorm=batchnorm, residual=residual)

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.transformer = TransformerEncoder(TransformerEncoderLayer(output_size, 2, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)
        self.bidirectional = bidirectional

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            z_params_res = self.residual['link'](x_res, z_prev)
        x = self.input_to_hidden(x)
        mask = None if self.bidirectional else self._generate_square_subsequent_mask(x.shape[-2])
        x = self.pe(x.transpose(-2, 0))
        outputs = self.transformer(x, mask=mask).transpose(-2, 0)

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        return z_params

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class CoattentiveTransformerLink(NamedLink):
    get_att = False

    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, n_targets=20, nheads=2,
                 sequence=None, memory=None, n_mems=None, mem_size=None, no_sa=False):
        super(CoattentiveTransformerLink, self).__init__(input_size, output_size, z_size, depth, params, embedding,
                                                         highway, dropout=dropout, batchnorm=batchnorm,
                                                         residual=residual)
        # assert output_size % n_targets == 0
        assert z_size % n_targets == 0
        # output_size = int(output_size/n_targets)
        self.target = nn.Embedding(n_targets, output_size).weight
        self.n_mems = n_mems
        self.memory = memory
        self.sequence = sequence
        self.att_vals = None

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.mem_to_hidden = nn.Linear(mem_size, output_size) if mem_size else None
        self.transformer_dec = TransformerDecoder(TransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size*n_targets,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.transformer_enc = TransformerEncoder(TransformerEncoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        if no_sa:
            for layer in self.transformer_dec.layers:
                layer.self_attn = IdentitySAPatch()
        # print("========Encoder Transformer Size========")
        # print("Enc params:",
        #       "output_size:", output_size,  "nheads:", nheads, "dim_feedforward:", output_size*n_targets, "depth:", depth)
        # print("Dec params:",
        #       "output_size:", output_size,  "nheads:", nheads, "dim_feedforward:", output_size, "depth:", depth)
        # number_parameters = sum(p.numel() for p in self.transformer_enc.parameters() if p.requires_grad)
        # print("TransEnc has {} M params".format(number_parameters/1e6))
        # number_parameters = sum(p.numel() for p in self.transformer_dec.parameters() if p.requires_grad)
        # print("TransDec has {} M params".format(number_parameters/1e6))
        # print("========================================")
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, int(z_size/n_targets))
                                                     for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.sequence is not None:
            if self.memory is not None:
                memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
                memory = memory.view((-1, self.n_mems, int(memory.shape[-1] / self.n_mems)))
                memory = self.mem_to_hidden(memory) if self.mem_to_hidden else memory
            x = torch.cat([v for k, v in x.items() if k in self.sequence], dim=-1)
            if self.residual is not None:
                x_res, x = x
                z_params_res = self.residual['link'](x_res, z_prev)

            x = self.input_to_hidden(x)
            if x.ndim > 3:
                batch_orig_shape = x.shape[:-2]
                x = x.view(-1, *x.shape[-2:])
            else:
                batch_orig_shape = None
            x = self.transformer_enc(self.pe(x.transpose(-2, 0)))
            seq_len = x.shape[0]
            if self.memory is not None:
                x = torch.cat([x, memory.transpose(0, -2)])
        else:
            if self.memory is not None:
                memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)
                seq_len = memory.shape[-2]
                if memory.ndim > 3:
                    batch_orig_shape = memory.shape[:-2]
                else:
                    batch_orig_shape = None
                memory = memory[..., 0, :]
                memory = memory.view((-1, self.n_mems, int(memory.shape[-1] / self.n_mems)))
                memory = self.mem_to_hidden(memory) if self.mem_to_hidden else memory
                x = memory.transpose(0, -2)
            else:
                raise LookupError('Memory and sequence are both None. You have to provide either of those.')
        target = self.target
        while target.ndim < x.ndim:
            new_dimension = x.ndim - target.ndim
            target = target.unsqueeze(1)
            target = target.expand((target.shape[0], x.shape[new_dimension], *target.shape[2:]))
        # This conditioned is not checked by the transformer module architecture
        assert all([ms == ts for ms, ts in zip(x.shape[1:], target.shape[1:])]), "Memory shape is {}, while  " \
                                                                                 "Target Shape is {}".format(x.shape,
                                                                                                             target.shape)
        outputs = self.transformer_dec(memory=x, tgt=target).transpose(-2, 0)
        if self.get_att:
            self.att_vals = []
            out = target
            for mod in self.transformer_dec.layers:
                self.att_vals.append(
                mod.multihead_attn(out, x, x)[1])
                out = mod(out, x)

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}

        z_params = {k: v.reshape(*v.shape[:-2], 1, v.shape[-2]*v.shape[-1]).expand(*v.shape[:-2], seq_len,
                                                                                   v.shape[-2]*v.shape[-1])
                    for k, v in z_params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']
        return z_params

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PoolingTransformerLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, nheads=4):
        super(PoolingTransformerLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                              dropout=dropout, batchnorm=batchnorm, residual=residual)

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.transformer = TransformerEncoder(TransformerEncoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)
        self.bidirectional = bidirectional

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.residual is not None:
            x_res, x = x
            z_params_res = self.residual['link'](x_res, z_prev)
        x = self.input_to_hidden(x)
        if x.ndim > 3:
            batch_orig_shape = x.shape[:-2]
            x = x.view(-1, *x.shape[-2:])
        else:
            batch_orig_shape = None
        mask = None if self.bidirectional else self._generate_square_subsequent_mask(x.shape[-2])
        x = self.pe(x.transpose(-2, 0))
        outputs = self.transformer(x, mask=mask).transpose(-2, 0)
        seq_len = outputs.shape[-2]
        outputs = outputs.mean(-2, keepdim=True).expand(*outputs.shape[:-2], seq_len, outputs.shape[-1])

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']

        return z_params


class TokenConditionedTransformerLink(NamedLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, nheads=4,
                 sequence=None, memory=None, mem_size=None):
        super(TokenConditionedTransformerLink, self).__init__(input_size, output_size, z_size, depth, params,
                                                              embedding, highway, dropout=dropout, batchnorm=batchnorm,
                                                              residual=residual)
        self.memory, self.sequence = memory, sequence
        assert memory is not None and sequence is not None
        self.mem_size = mem_size
        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.mem_to_hidden = nn.Linear(mem_size, output_size)
        self.transformer = TransformerEncoder(TransformerEncoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)
        self.bidirectional = bidirectional

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)
        memory = self.mem_to_hidden(memory)
        x = torch.cat([v for k, v in x.items() if k in self.sequence], dim=-1)

        x = self.input_to_hidden(x)
        # Adding Token conditioning
        x = torch.cat([memory[..., :1, :], x], dim=-2)
        if x.ndim > 3:
            batch_orig_shape = x.shape[:-2]
            x = x.view(-1, *x.shape[-2:])
        else:
            batch_orig_shape = None

        mask = None if self.bidirectional else self._generate_square_subsequent_mask(x.shape[-2])
        x = self.pe(x.transpose(-2, 0))
        outputs = self.transformer(x, mask=mask)[1:].transpose(-2, 0)

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        return z_params

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class CoattentiveBARTTransformerLink(NamedLink):
    get_att = False

    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, dropout=0., residual=None,
                 n_targets=20, n_mems=None, fr=False):
        super(CoattentiveBARTTransformerLink, self).__init__(input_size, output_size, z_size, depth, params, embedding,
                                                         highway=False, dropout=dropout, batchnorm=False,
                                                         residual=residual)
        # assert output_size % n_targets == 0
        assert z_size % n_targets == 0
        self.transformer = AutoModel.from_pretrained(BARTHEZ_LINK, local_files_only=LOCAL_ONLY) if fr else \
            BartModel.from_pretrained(BART_LINK, local_files_only=LOCAL_ONLY)
        assert output_size == self.transformer.config.d_model
        output_size = self.transformer.config.d_model
        # output_size = int(output_size/n_targets)
        self.target = nn.Embedding(n_targets, output_size).weight
        self.n_mems = n_mems
        self.att_vals = None

        self.input_to_hidden = nn.Linear(input_size, output_size)

        if embedding is not None:
            z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, int(z_size/n_targets))
                                                     for param in params})

    def forward(self, x, z_prev=None, lens=None):
        x = torch.cat([v for k, v in x.items()], dim=-1)
        if self.residual is not None:
            x_res, x = x
            z_params_res = self.residual['link'](x_res, z_prev)

        x = self.input_to_hidden(x)
        if x.ndim > 3:
            batch_orig_shape = x.shape[:-2]
            x = x.view(-1, *x.shape[-2:])
        else:
            batch_orig_shape = None

        seq_len = x.shape[1]
        target = self.target
        while target.ndim < x.ndim:
            new_dimension = x.ndim - target.ndim-1
            target = target.unsqueeze(0)
            target = target.expand((x.shape[new_dimension], target.shape[1], *target.shape[2:]))
        # This conditioned is not checked by the transformer module architecture
        # assert all([ms == ts for ms, ts in zip(x.shape[0:], target.shape[0:])]), "Memory shape is {}, while  " \
        #                                                                          "Target Shape is {}".format(x.shape,
        #                                                                                                      target.shape)
        outputs = self.transformer(inputs_embeds=x, decoder_inputs_embeds=target, output_attentions=self.get_att)
        if self.get_att:
            self.att_vals = [att.mean(1) for att in outputs.cross_attentions]
        outputs = outputs.last_hidden_state

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}

        z_params = {k: v.reshape(*v.shape[:-2], 1, v.shape[-2]*v.shape[-1]).expand(*v.shape[:-2], seq_len,
                                                                                   v.shape[-2]*v.shape[-1])
                    for k, v in z_params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']
        return z_params


class ConditionalCoattentiveBARTTransformerLink(NamedLink):
    get_att = False

    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., bidirectional=False, n_mems=20, memory=None, targets=None,
                 mem_size=None, fr=False):
        super(ConditionalCoattentiveBARTTransformerLink, self).__init__(input_size, output_size, z_size, depth,
                                                                    params, embedding, highway, dropout=dropout,
                                                                    batchnorm=False, residual=None)
        self.transformer = AutoModel.from_pretrained(BARTHEZ_LINK, local_files_only=LOCAL_ONLY) if fr else \
            BartModel.from_pretrained(BART_LINK, local_files_only=LOCAL_ONLY)
        assert output_size == self.transformer.config.d_model
        output_size = self.transformer.config.d_model

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.mem_size = mem_size or int(input_size/n_mems)
        self.memory_to_hidden = nn.Linear(self.mem_size, output_size)

        self.memory, self.targets = memory, targets
        self.n_mems, self.output_size = n_mems, output_size
        self.bidirectional = bidirectional
        self.att_vals = None

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

    def forward(self, x, z_prev=None, lens=None):
        memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
        memory = memory.view((*memory.shape[:-1], self.n_mems, self.mem_size))
        memory = self.memory_to_hidden(memory)
        targets = torch.cat([v for k, v in x.items() if k in self.targets], dim=-1)

        if memory.ndim > 3:
            batch_orig_shape = memory.shape[:-2]
            memory = memory.view(-1, *memory.shape[-2:])
            targets = targets.view(-1, *targets.shape[-2:])
        else:
            batch_orig_shape = None
        targets = self.input_to_hidden(targets)

        # This conditioned is not checked by the transformer module architecture
        # assert all([ms == ts for ms, ts in zip(memory.shape[0:], targets.shape[0:])])
        outputs = self.transformer(inputs_embeds=memory, decoder_inputs_embeds=targets, output_attentions=self.get_att)
        if self.get_att:
            self.att_vals = [att.mean(1) for att in outputs.cross_attentions]
        outputs = outputs.last_hidden_state

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}

        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))

        return z_params


class QKVBartTransformerLink(NamedLink):
    get_att = False

    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, n_mems=20, n_keys=1, memory=None,
                 key=None, targets=None, nheads=2, minimal_enc=False, mem_size=None, old_ver=False,
                 simple_zs_use=True, layer_wise=False, fr=False):
        super(QKVBartTransformerLink, self).__init__(input_size, output_size, z_size, depth,
                                                                    params, embedding, highway, dropout=dropout,
                                                                    batchnorm=batchnorm, residual=residual)
        self.transformer_dec = AutoModel.from_pretrained(BARTHEZ_LINK, local_files_only=LOCAL_ONLY).decoder if fr else \
            BartModel.from_pretrained(BART_LINK, local_files_only=LOCAL_ONLY).decoder
        assert output_size == self.transformer_dec.config.d_model, "Output size {}, different from BART model " \
                                                                   "dimension {}" \
                                                                   "".format(output_size,
                                                                             self.transformer_dec.config.d_model)
        self.n_layers = len(self.transformer_dec.layers) if layer_wise else 0
        output_size = self.transformer_dec.config.d_model
        hack_BART(self.transformer_dec)

        # output_size = int(output_size/n_mems)
        self.mem_ids = nn.Embedding(n_mems, mem_size).weight
        self.mem_size = mem_size or int(input_size/n_mems)
        self.simple_zs_use = simple_zs_use
        self.n_keys = n_mems if simple_zs_use else n_keys
        self.memory_to_hidden = nn.Linear(self.mem_size*2, output_size * (self.n_layers or 1))
        self.old = old_ver
        if self.old:
            self.key_to_hidden = nn.Linear(self.mem_size, output_size * self.n_keys)
        else:
            nn.Sequential(*([nn.Linear(self.mem_size, output_size * self.n_keys),
                             nn.GELU()] * depth)[:-1])
            k2h_layers = []
            if depth > 1:
                k2h_layers.append(nn.Linear(self.mem_size * n_mems, output_size))
                k2h_layers.append(nn.GELU())
                for _ in range(depth-2):
                    k2h_layers.append(nn.Linear(output_size, output_size))
                    k2h_layers.append(nn.GELU())

                k2h_layers.append(nn.Linear(output_size, output_size * self.n_keys * (self.n_layers or 1)))
            else:
                if output_size < self.mem_size * n_mems:  # to minimize this layer's size
                    k2h_layers.append(nn.Linear(self.mem_size * n_mems, output_size))
                    k2h_layers.append(nn.Linear(output_size, output_size * self.n_keys * (self.n_layers or 1)))
                else:
                    k2h_layers.append(nn.Linear(self.mem_size * n_mems,
                                                output_size * self.n_keys * (self.n_layers or 1)))

            self.key_to_hidden = nn.Sequential(*k2h_layers)

        if not self.simple_zs_use:
            self.key_inputs = nn.Embedding(n_mems, output_size).weight
            self.key_enc = TransformerDecoder(TransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        assert (memory is None and key is None) or (memory is not None and key is not None), "if you specify memory" \
                                                                                             " variables, also specify" \
                                                                                             " key variables"
        self.memory, self.key, self.targets = memory, key, targets
        self.bn = nn.BatchNorm1d(z_size)
        self.n_mems, self.output_size = n_mems, output_size
        self.bidirectional = bidirectional
        self.att_vals = None

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

        assert self.residual is None, "Named links still can't have residuals"

    def forward(self, x, z_prev=None, lens=None):
        memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
        memory = memory.view((*memory.shape[:-1], self.n_mems, self.mem_size))
        mem_ids = self.mem_ids
        while memory.ndim > mem_ids.ndim:
            mem_ids = mem_ids.unsqueeze(0)
        memory = self.memory_to_hidden(torch.cat([memory, self.mem_ids.expand(memory.shape)], dim=-1))
        key = torch.cat([v for k, v in x.items() if k in self.key], dim=-1)[..., 0, :]
        if self.old:
            key = key.view((*key.shape[:-1], 1, self.mem_size))
        else:
            key = key.view((*key.shape[:-1], self.mem_size*self.n_mems))
        key = self.key_to_hidden(key).view((*key.shape[:-1], self.n_keys, self.output_size * (self.n_layers or 1)))
        targets = torch.cat([v for k, v in x.items() if k in self.targets], dim=-1)

        if memory.ndim > 3:
            batch_orig_shape = memory.shape[:-2]
            memory = memory.view(-1, *memory.shape[-2:])
            key = key.view(-1, *key.shape[-2:])
            targets = targets.view(-1, *targets.shape[-2:])
        else:
            batch_orig_shape = None

        if not self.simple_zs_use:
            key = key.transpose(-2, 0)
            key_inputs = self.key_inputs.unsqueeze(1).expand(self.key_inputs.shape[0], key.shape[-2],
                                                             self.key_inputs.shape[1])
            key = self.key_enc(tgt=key_inputs, memory=key)
            key = key.transpose(-2, 0)

        if self.n_layers > 0:
            key = key.view(*key.shape[:-1], self.n_layers, int(key.shape[-1]/self.n_layers)).unbind(-2)
            memory = memory.view(*memory.shape[:-1], self.n_layers, int(memory.shape[-1]/self.n_layers)).unbind(-2)
            load_BART_kv_hacks(self.transformer_dec, key, memory)
        else:
            n_layers = len(self.transformer_dec.layers)
            load_BART_kv_hacks(self.transformer_dec, [key]*n_layers, [memory]*n_layers)
        outputs = self.transformer_dec(inputs_embeds=targets, encoder_hidden_states=memory,
                                       output_attentions=self.get_att)
        clear_BART_kv_hacks(self.transformer_dec)
        if self.get_att:
            self.att_vals = [att.mean(1) for att in outputs.cross_attentions]
        outputs = outputs.last_hidden_state

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}

        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        return z_params


class CoattentiveTransformerLink2(NamedLink):
    # This one was made with modifications for the QKV project that don't affect the previous ADVAE project
    get_att = False

    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, n_targets=20, nheads=2,
                 sequence=None, memory=None, n_mems=None, mem_size=None):
        super(CoattentiveTransformerLink2, self).__init__(input_size, output_size, z_size, depth, params, embedding,
                                                         highway, dropout=dropout, batchnorm=batchnorm,
                                                         residual=residual)
        # assert output_size % n_targets == 0
        assert z_size % n_targets == 0
        # output_size = int(output_size/n_targets)
        self.target = nn.Embedding(n_targets, output_size).weight
        self.n_mems = n_mems
        self.memory = memory
        self.sequence = sequence
        self.att_vals = None

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.mem_to_hidden = nn.Linear(mem_size, output_size) if mem_size else None
        self.transformer_dec = TransformerDecoder(TransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.transformer_enc = TransformerEncoder(TransformerEncoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, int(z_size/n_targets))
                                                     for param in params})

    def forward(self, x, z_prev=None, lens=None):
        if self.sequence is not None:
            if self.memory is not None:
                memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
                memory = memory.view((-1, self.n_mems, int(memory.shape[-1] / self.n_mems)))
                memory = self.mem_to_hidden(memory) if self.mem_to_hidden else memory
            x = torch.cat([v for k, v in x.items() if k in self.sequence], dim=-1)
            if self.residual is not None:
                x_res, x = x
                z_params_res = self.residual['link'](x_res, z_prev)

            x = self.input_to_hidden(x)
            if x.ndim > 3:
                batch_orig_shape = x.shape[:-2]
                x = x.view(-1, *x.shape[-2:])
            else:
                batch_orig_shape = None
            x = self.transformer_enc(self.pe(x.transpose(-2, 0)))
            seq_len = x.shape[0]
            if self.memory is not None:
                x = torch.cat([x, memory.transpose(0, -2)])
        else:
            if self.memory is not None:
                memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)
                seq_len = memory.shape[-2]
                if memory.ndim > 3:
                    batch_orig_shape = memory.shape[:-2]
                else:
                    batch_orig_shape = None
                memory = memory[..., 0, :]
                memory = memory.view((-1, self.n_mems, int(memory.shape[-1] / self.n_mems)))
                memory = self.mem_to_hidden(memory) if self.mem_to_hidden else memory
                x = memory.transpose(0, -2)
            else:
                raise LookupError('Memory and sequence are both None. You have to provide either of those.')
        target = self.target
        while target.ndim < x.ndim:
            new_dimension = x.ndim - target.ndim
            target = target.unsqueeze(1)
            target = target.expand((target.shape[0], x.shape[new_dimension], *target.shape[2:]))
        # This conditioned is not checked by the transformer module architecture
        assert all([ms == ts for ms, ts in zip(x.shape[1:], target.shape[1:])]), "Memory shape is {}, while  " \
                                                                                 "Target Shape is {}".format(x.shape,
                                                                                                             target.shape)
        outputs = self.transformer_dec(memory=x, tgt=target).transpose(-2, 0)
        if self.get_att:
            self.att_vals = []
            out = target
            for mod in self.transformer_dec.layers:
                self.att_vals.append(
                mod.multihead_attn(out, x, x)[1])
                out = mod(out, x)

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}

        z_params = {k: v.reshape(*v.shape[:-2], 1, v.shape[-2]*v.shape[-1]).expand(*v.shape[:-2], seq_len,
                                                                                   v.shape[-2]*v.shape[-1])
                    for k, v in z_params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        if self.residual is not None:
            z_params['loc'] = z_params_res['loc'] + z_params['loc']
        return z_params

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ConditionalCoattentiveTransformerLink(NamedLink):
    get_att = False

    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, n_mems=20, memory=None, targets=None,
                 nheads=2, minimal_enc=False, mem_size=None, mem_enc=True):
        super(ConditionalCoattentiveTransformerLink, self).__init__(input_size, output_size, z_size, depth,
                                                                    params, embedding, highway, dropout=dropout,
                                                                    batchnorm=batchnorm, residual=residual)
        # output_size = int(output_size/n_mems)

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.mem_size = mem_size or int(output_size/n_mems)
        self.memory_to_hidden = nn.Linear(self.mem_size, output_size)
        if mem_enc:
            if minimal_enc:
                self.transformer_enc = MinimalTransformerEncoder(output_size, n_mems)
            else:
                self.transformer_enc = TransformerEncoder(SpecialTransformerEncoder(output_size, nheads, dim_feedforward=output_size,
                                                                                    dropout=dropout, activation='gelu',
                                                                                    n_mems=n_mems), depth)
        else:
            self.transformer_enc = None
        self.transformer_dec = TransformerDecoder(TransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)

        # print("========Decoder Transformer Size========")
        # print("Dec params:",
        #       "output_size:", output_size,  "nheads:", nheads, "dim_feedforward:", output_size, "depth:", depth)
        # number_parameters = sum(p.numel() for p in self.transformer_dec.parameters() if p.requires_grad)
        # print("TransDec has {} M params".format(number_parameters/1e6))
        # print("========================================")
        self.memory, self.targets = memory, targets
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)
        self.n_mems, self.output_size = n_mems, output_size
        self.bidirectional = bidirectional
        self.att_vals = None

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})
        assert self.residual is None, "Named links still can't have residuals"

    def forward(self, x, z_prev=None, lens=None):
        memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
        memory = memory.view((*memory.shape[:-1], self.n_mems, self.mem_size))
        memory = self.memory_to_hidden(memory)
        targets = torch.cat([v for k, v in x.items() if k in self.targets], dim=-1)

        if memory.ndim > 3:
            batch_orig_shape = memory.shape[:-2]
            memory = memory.view(-1, *memory.shape[-2:])
            targets = targets.view(-1, *targets.shape[-2:])
        else:
            batch_orig_shape = None
        targets = self.input_to_hidden(targets)
        targets = self.pe(targets.transpose(-2, 0))
        target_mask = self._generate_square_subsequent_mask(targets.shape[0]) if not self.bidirectional else None
        # memory = self.pe(memory.transpose(-2, 0))
        memory = memory.transpose(-2, 0)
        memory = self.transformer_enc(memory) if self.transformer_enc is not None else memory

        # This conditioned is not checked by the transformer module architecture
        assert all([ms == ts for ms, ts in zip(memory.shape[1:], targets.shape[1:])])
        outputs = self.transformer_dec(memory=memory, tgt=targets, tgt_mask=target_mask).transpose(-2, 0)

        if self.get_att:
            self.att_vals = []
            out = targets
            for mod in self.transformer_dec.layers:
                self.att_vals.append(
                mod.multihead_attn(out, memory, memory)[1])
                out = mod(out, x)

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}

        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        return z_params

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ConditionalCoattentiveQKVTransformerLink(NamedLink):
    get_att = False

    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, n_mems=20, n_keys=1, memory=None,
                 key=None, targets=None, nheads=2, minimal_enc=False, mem_size=None, old_ver=False,
                 simple_zs_use=True):
        super(ConditionalCoattentiveQKVTransformerLink, self).__init__(input_size, output_size, z_size, depth,
                                                                    params, embedding, highway, dropout=dropout,
                                                                    batchnorm=batchnorm, residual=residual)
        # output_size = int(output_size/n_mems)
        self.mem_ids = nn.Embedding(n_mems, mem_size).weight
        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.mem_size = mem_size or int(output_size/n_mems)
        self.simple_zs_use = simple_zs_use
        self.n_keys = n_mems if simple_zs_use else n_keys
        self.memory_to_hidden = nn.Linear(self.mem_size*2, output_size)
        self.old = old_ver
        if self.old:
            self.key_to_hidden = nn.Linear(self.mem_size, output_size * self.n_keys)
        else:
            nn.Sequential(*([nn.Linear(self.mem_size, output_size * self.n_keys),
                             nn.GELU()] * depth)[:-1])
            k2h_layers = []
            if depth>1:
                k2h_layers.append(nn.Linear(self.mem_size * n_mems, output_size))
                k2h_layers.append(nn.GELU())
                for _ in range(depth-2):
                    k2h_layers.append(nn.Linear(output_size, output_size))
                    k2h_layers.append(nn.GELU())

                k2h_layers.append(nn.Linear(output_size, output_size * self.n_keys))
            else:
                if output_size < self.mem_size * n_mems:  # to minimize this layer's size
                    k2h_layers.append(nn.Linear(self.mem_size * n_mems, output_size))
                    k2h_layers.append(nn.Linear(output_size, output_size * self.n_keys))
                else:
                    k2h_layers.append(nn.Linear(self.mem_size * n_mems, output_size * self.n_keys))

            self.key_to_hidden = nn.Sequential(*k2h_layers)

        if not self.simple_zs_use:
            self.key_inputs = nn.Embedding(n_mems, output_size).weight
            self.key_enc = TransformerDecoder(TransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.transformer_dec = QKVTransformerDecoder(
            QKVTransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size, dropout=dropout,
                                       activation='gelu'), depth)
        assert (memory is None and key is None) or (memory is not None and key is not None), "if you specify memory" \
                                                                                             " variables, also specify" \
                                                                                             " key variables"
        self.memory, self.key, self.targets = memory, key, targets
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)
        self.n_mems, self.output_size = n_mems, output_size
        self.bidirectional = bidirectional
        self.att_vals = None

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_size) for param in params})

        assert self.residual is None, "Named links still can't have residuals"

    def forward(self, x, z_prev=None, lens=None):
        memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
        memory = memory.view((*memory.shape[:-1], self.n_mems, self.mem_size))
        mem_ids = self.mem_ids
        while memory.ndim > mem_ids.ndim:
            mem_ids = mem_ids.unsqueeze(0)
        memory = self.memory_to_hidden(torch.cat([memory, self.mem_ids.expand(memory.shape)], dim=-1))
        key = torch.cat([v for k, v in x.items() if k in self.key], dim=-1)[..., 0, :]
        if self.old:
            key = key.view((*key.shape[:-1], 1, self.mem_size))
        else:
            key = key.view((*key.shape[:-1], self.mem_size*self.n_mems))
        key = self.key_to_hidden(key).view((*key.shape[:-1], self.n_keys, self.output_size))
        targets = torch.cat([v for k, v in x.items() if k in self.targets], dim=-1)

        if memory.ndim > 3:
            batch_orig_shape = memory.shape[:-2]
            memory = memory.view(-1, *memory.shape[-2:])
            key = key.view(-1, *key.shape[-2:])
            targets = targets.view(-1, *targets.shape[-2:])
        else:
            batch_orig_shape = None
        targets = self.input_to_hidden(targets)
        targets = self.pe(targets.transpose(-2, 0))
        target_mask = self._generate_square_subsequent_mask(targets.shape[0]) if not self.bidirectional else None
        # memory = self.pe(memory.transpose(-2, 0))
        memory = memory.transpose(-2, 0)
        # memory = self.transformer_enc(memory)
        key = key.transpose(-2, 0)
        if not self.simple_zs_use:
            key_inputs = self.key_inputs.unsqueeze(1).expand(self.key_inputs.shape[0], key.shape[-2],
                                                             self.key_inputs.shape[1])
            key = self.key_enc(tgt=key_inputs, memory=key)

        # This conditioned is not checked by the transformer module architecture
        assert all([ms == ts for ms, ts in zip(memory.shape[1:], targets.shape[1:])])
        outputs = self.transformer_dec(memory=memory, tgt=targets, key=key, tgt_mask=target_mask).transpose(-2, 0)

        if self.get_att:
            self.att_vals = []
            out = targets
            for mod in self.transformer_dec.layers:
                self.att_vals.append(
                mod.multihead_attn(out, memory, memory)[1])
                out = mod(out, x)

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}

        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        return z_params

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class ConditionalCoattentiveTransformerLink2(NamedLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, sn_mems=20, tn_mems=20, memory=None,
                 targets=None, nheads=2, minimal_enc=False):
        super(ConditionalCoattentiveTransformerLink2, self).__init__(input_size, output_size, z_size, depth,
                                                                    params, embedding, highway, dropout=dropout,
                                                                    batchnorm=batchnorm, residual=residual)
        output_size = int(output_size/tn_mems)

        if memory is not None:
            if minimal_enc:
                self.transformer_enc = MinimalTransformerEncoder(output_size, sn_mems)
            else:
                self.transformer_enc = TransformerEncoder(SpecialTransformerEncoder(output_size, nheads,
                                                                                    dim_feedforward=output_size*sn_mems,
                                                                                    dropout=dropout, activation='gelu',
                                                                                    n_mems=sn_mems), depth)
            self.transformer_dec = TransformerDecoder(TransformerDecoderLayer(output_size, nheads,
                                                                              dim_feedforward=output_size,
                                                                              dropout=dropout, activation='gelu'), depth)
        else:
            self.transformer_dec = TransformerEncoder(SpecialTransformerEncoder(output_size, nheads,
                                                                                dim_feedforward=output_size,
                                                                                dropout=dropout, activation='gelu',
                                                                                n_mems=tn_mems), depth)

        self.memory, self.targets = memory, targets
        self.pe = MinimalTransformerEncoder(output_size, tn_mems)
        self.bn = nn.BatchNorm1d(z_size)
        self.sn_mems, self.tn_mems, self.output_size = sn_mems, tn_mems, output_size
        self.bidirectional = bidirectional

        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size, int(z_size/tn_mems))
                                                     for param in params})
        assert self.residual is None, "Named links still can't have residuals"

    def forward(self, x, z_prev=None, lens=None):
        targets = torch.cat([v for k, v in x.items() if k in self.targets], dim=-1)
        seq_len = targets.shape[-2]
        targets = targets[..., 0, :]
        targets = targets.view((*targets.shape[:-1], self.tn_mems, self.output_size))

        if targets.ndim > 3:
            batch_orig_shape = targets.shape[:-2]
            targets = targets.view(-1, *targets.shape[-2:])
        else:
            batch_orig_shape = None

        targets = self.pe(targets.transpose(-2, 0))
        target_mask = self._generate_square_subsequent_mask(targets.shape[0]) if not self.bidirectional else None
        if self.memory is not None:
            memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
            memory = memory.view((*memory.shape[:-1], self.sn_mems, self.output_size))
            if batch_orig_shape is not None:
                memory = memory.view(-1, *memory.shape[-2:])
                targets = targets.view(-1, *targets.shape[-2:])

            memory = memory.transpose(-2, 0)
            memory = self.transformer_enc(memory)

            # This conditioned is not checked by the transformer module architecture
            assert all([ms == ts for ms, ts in zip(memory.shape[1:], targets.shape[1:])])
            outputs = self.transformer_dec(memory=memory, tgt=targets, tgt_mask=target_mask).transpose(-2, 0)
        else:
            outputs = self.transformer_dec(targets, mask=target_mask).transpose(-2, 0)
        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        z_params = {k: v.reshape(*v.shape[:-2], 1, v.shape[-2] * v.shape[-1]).expand(*v.shape[:-2], seq_len,
                                                                                     v.shape[-2] * v.shape[-1])
                    for k, v in z_params.items()}

        if batch_orig_shape is not None:
            z_params = {k: v.view((*batch_orig_shape, *v.shape[-2:])) for k, v in z_params.items()}

        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        if 'loc' in z_params and self.batchnorm:
            out_shape = z_params['loc'].shape
            z_params['loc'] = self.bn(z_params['loc'].view(-1, out_shape[-1])).view(out_shape)
            out_shape = z_params['scale'].shape
            z_params['scale'] = self.bn(z_params['scale'].view(-1, out_shape[-1]).log()
                                             ).view(out_shape).exp()

        return z_params

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    # Took this snippet from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SpecialTransformerEncoder(TransformerEncoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", n_mems=20):
        super(SpecialTransformerEncoder, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.k, self.q, self.v = nn.Embedding(n_mems, int(d_model/2)).weight, nn.Embedding(n_mems, int(d_model/2)).weight,\
                                 nn.Embedding(n_mems, int(d_model/2)).weight
        self.linear0 = torch.nn.Linear(d_model, int(d_model/2))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        q = self.q.unsqueeze(1).expand(self.q.shape[0], src.shape[1], self.q.shape[1])
        k = self.k.unsqueeze(1).expand(self.k.shape[0], src.shape[1], self.k.shape[1])
        v = self.v.unsqueeze(1).expand(self.v.shape[0], src.shape[1], self.v.shape[1])
        src1 = self.linear0(src)
        src2 = self.self_attn(torch.cat([src1, q], -1), torch.cat([src1, k], -1), torch.cat([src1, v], -1),
                              attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class QKVTransformerDecoderLayer(TransformerDecoderLayer):

    def forward(self, tgt, memory, key, tgt_mask = None, memory_mask= None,
                tgt_key_padding_mask= None, memory_key_padding_mask= None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, key, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class QKVTransformerDecoder(TransformerDecoder):

    def forward(self, tgt, memory, key, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, key, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MinimalTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_mems=20):
        super(MinimalTransformerEncoder, self).__init__()
        self.embs = nn.Embedding(n_mems, d_model).weight

    def forward(self, src):
        embs = self.embs.unsqueeze(1).expand(self.embs.shape[0], src.shape[1], self.embs.shape[1])
        return src+embs


class HackedBARTAttention(BartAttention):
    def __init__(self, bart_att_obj):
        super(HackedBARTAttention, self).__init__(bart_att_obj.embed_dim, bart_att_obj.num_heads)
        self.embed_dim = bart_att_obj.embed_dim
        self.num_heads = bart_att_obj.num_heads
        self.dropout = bart_att_obj.dropout
        self.head_dim = bart_att_obj.head_dim

        self.scaling = bart_att_obj.scaling
        self.is_decoder = bart_att_obj.is_decoder 

        self.k_proj = bart_att_obj.k_proj
        self.v_proj = bart_att_obj.v_proj
        self.q_proj = bart_att_obj.q_proj
        self.out_proj = bart_att_obj.out_proj
        self.k_v_hack = None

    def load_kv_hack(self, keys, values):
        self.k_v_hack = (keys, values)

    def clear_kv_hack(self):
        self.k_v_hack = None

    def forward(self, hidden_states, key_value_states, past_key_value, attention_mask,
                layer_head_mask, output_attentions):
        assert self.k_v_hack is not None
        return super(HackedBARTAttention, self).forward(hidden_states, self.k_v_hack[0], self.k_v_hack, attention_mask,
                                                 layer_head_mask, output_attentions)

def hack_BART(bart_decoder):
    for layer in bart_decoder.layers:
        layer.encoder_attn = HackedBARTAttention(layer.encoder_attn)

def load_BART_kv_hacks(bart_decoder, k, v):
    for layer, k_i, v_i in zip(bart_decoder.layers, k, v):
        layer.encoder_attn.load_kv_hack(k_i.contiguous(), v_i.contiguous())

def clear_BART_kv_hacks(bart_decoder):
    for layer in bart_decoder.layers:
        layer.encoder_attn.clear_kv_hack()


class IdentitySAPatch(nn.Module):
    def __init__(self):
        super(IdentitySAPatch, self).__init__()

    def forward(self, tgt1, tgt2, tgt3, attn_mask=None, key_padding_mask=None):
                return tgt1, None

