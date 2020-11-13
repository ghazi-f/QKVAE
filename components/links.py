import math
import abc
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, \
    TransformerDecoderLayer
import torch.nn.functional as F

EPSILON = 1e-8


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

        packed_outputs, (hidden, cell) = self.rnn(packed_x)

        if self.last_state:
            outputs = torch.cat([hidden[-1, :, :], hidden[-2, :, :]] if self.bidirectional else
                                [hidden[-1, :, :]], dim=1)
            outputs = outputs.unsqueeze(1).repeat(1, x.shape[0], 1)
        else:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=x.shape[0])
        if batch_shape is not None:
            outputs = outputs.view(*batch_shape, *outputs.shape[-2:])

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
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, n_targets=20, nheads=2,
                 sequence=None, memory=None, n_mems=None):
        super(CoattentiveTransformerLink, self).__init__(input_size, output_size, z_size, depth, params, embedding,
                                                         highway, dropout=dropout, batchnorm=batchnorm,
                                                         residual=residual)
        assert output_size % n_targets == 0
        assert z_size % n_targets == 0
        output_size = int(output_size/n_targets)
        self.target = nn.Embedding(n_targets, output_size).weight
        self.n_mems = n_mems
        self.memory = memory
        self.sequence = sequence

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.transformer_dec = TransformerDecoder(TransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size*n_targets,
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
                x = memory.transpose(0, -2)
            else:
                raise LookupError('Memory and sequence are both None. You have to provide either of those.')
        target = self.target
        while target.ndim < x.ndim:
            new_dimension = x.ndim - target.ndim
            target = target.unsqueeze(1)
            target = target.expand((target.shape[0], x.shape[new_dimension], *target.shape[2:]))
        # This conditioned is not checked by the transformer module architecture
        assert all([ms == ts for ms, ts in zip(x.shape[1:], target.shape[1:])])
        outputs = self.transformer_dec(memory=x, tgt=target).transpose(-2, 0)

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
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None, bidirectional=False, n_mems=20, memory=None, targets=None,
                 nheads=2):
        super(ConditionalCoattentiveTransformerLink, self).__init__(input_size, output_size, z_size, depth,
                                                                    params, embedding, highway, dropout=dropout,
                                                                    batchnorm=batchnorm, residual=residual)
        output_size = int(output_size/n_mems)

        self.input_to_hidden = nn.Linear(input_size, output_size)
        self.transformer_enc = TransformerEncoder(SpecialTransformerEncoder(output_size, nheads, dim_feedforward=output_size*n_mems,
                                                                      dropout=dropout, activation='gelu', n_mems=n_mems)
                                                  , depth)
        self.transformer_dec = TransformerDecoder(TransformerDecoderLayer(output_size, nheads, dim_feedforward=output_size,
                                                                      dropout=dropout, activation='gelu'), depth)
        self.memory, self.targets = memory, targets
        self.pe = PositionalEncoding(output_size)
        self.bn = nn.BatchNorm1d(z_size)
        self.n_mems, self.output_size = n_mems, output_size
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
        assert self.residual is None, "Named links still can't have residuals"

    def forward(self, x, z_prev=None, lens=None):
        memory = torch.cat([v for k, v in x.items() if k in self.memory], dim=-1)[..., 0, :]
        memory = memory.view((*memory.shape[:-1], self.n_mems, self.output_size))
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
        memory = self.transformer_enc(memory)

        # This conditioned is not checked by the transformer module architecture
        assert all([ms == ts for ms, ts in zip(memory.shape[1:], targets.shape[1:])])
        outputs = self.transformer_dec(memory=memory, tgt=targets, tgt_mask=target_mask).transpose(-2, 0)

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
