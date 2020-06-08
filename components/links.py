import math
import abc
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
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


# ============================================== LINK CLASSES ==========================================================


class MLPLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0., batchnorm=False, residual=None):
        super(MLPLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout, batchnorm=batchnorm, residual=residual)

        self.mlp = nn.ModuleList([nn.Linear(input_size, output_size)] +
                                 [nn.Linear((input_size + output_size*i) if self.highway else output_size, output_size)
                                  for i in range(1, depth)])
        self.drp_layer = torch.nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(z_size)

        mlp_out_size = (output_size * depth) if self.highway else output_size
        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_params_size)
                                                     for param in params})
            #self.to_emb = nn.Linear(z_params_size, embedding.weight.shape[0])
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size) for param in params})

    def forward(self, x, z_prev=None):
        if self.residual is not None:
            x_res, x = x
            z_params_res = self.residual['link'](x_res, z_prev)

        outputs = []
        input = x
        for layer in self.mlp:
            outputs.append(layer(F.gelu(input)))# if len(outputs) else x)
            if self.dropout > 0 and 'loc' in self.hidden_to_z_params:
                outputs[-1] = self.drp_layer(outputs[-1])
            input = torch.cat([input, outputs[-1]], dim=-1) if self.highway else outputs[-1]

        outputs = torch.cat(outputs, dim=-1) if self.highway else outputs[-1]

        z_params = {param: activation(self.hidden_to_z_params[param](outputs))+EPSILON for param, activation in
                    self.params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
                #z_params['logits'] = self.to_emb(z_params['logits'])
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

    def forward(self, x, z_prev=None):
        if self.residual is not None:
            x_res, x = x
            z_params_res = self.residual['link'](x_res, z_prev)
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

        if self.dropout > 0.:
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

    def forward(self, x, z_prev=None):
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
