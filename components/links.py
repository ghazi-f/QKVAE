import math
import abc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================== BASE CLASSES ==========================================================

class BaseLink(nn.Module):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, dropout=0.):
        super(BaseLink, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.z_size = z_size
        self.depth = depth
        self.params = params
        self.embedding = embedding
        self.highway = highway
        self.dropout = dropout


class SequentialLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, dropout=0.):
        super(SequentialLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                             dropout=dropout)


# ============================================== LINK CLASSES ==========================================================

class MLPLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0.):
        super(MLPLink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout)

        self.mlp = nn.ModuleList([nn.Linear(input_size, output_size)] +
                                 [nn.Linear((input_size + output_size*i) if self.highway else output_size, output_size)
                                  for i in range(1, depth)])
        self.drp_layer = torch.nn.Dropout(p=dropout)

        mlp_out_size = (output_size * depth) if self.highway else output_size
        if embedding is not None:
            self.sbn = sbn
            if sbn is not None:
                z_params_size = int(embedding.weight.shape[1] / sbn.n_experts)
            else:
                z_params_size = embedding.weight.shape[1]
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_params_size)
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(mlp_out_size, z_size) for param in params})

    def forward(self, x, z_prev=None):
        outputs = []
        input = x
        for layer in self.mlp:
            outputs.append(layer(F.gelu(input)))# if len(outputs) else x)
            if self.dropout >0:
                outputs[-1] = self.drp_layer(outputs[-1])
            input = torch.cat([input, outputs[-1]], dim=-1) if self.highway else outputs[-1]

        outputs = torch.cat(outputs, dim=-1) if self.highway else outputs[-1]
        z_params = {param: activation(self.hidden_to_z_params[param](outputs)) for param, activation in
                    self.params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))

        return z_params


class GRULink(SequentialLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None, highway=False, sbn=None,
                 dropout=0.):
        super(GRULink, self).__init__(input_size, output_size, z_size, depth, params, embedding, highway,
                                      dropout=dropout)
        self.rnn = nn.GRU(input_size, output_size, depth, batch_first=True, dropout=dropout)
        self.drp_layer = torch.nn.Dropout(p=dropout)
        rnn_output_size = (output_size*depth) if highway else output_size
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
        z_params = {param: activation(self.hidden_to_z_params[param](reshaped_h))
                    for param, activation in self.params.items()}
        if self.embedding is not None:
            if self.sbn is not None:
                z_params['logits'] = self.sbn(z_params['logits'], self.embedding.weight)
            else:
                z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))

        return z_params
