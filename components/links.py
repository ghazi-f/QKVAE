import math
import abc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================== BASE CLASSES ==========================================================

class BaseLink(nn.Module):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None):
        super(BaseLink, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.z_size = z_size
        self.depth = depth
        self.params = params
        self.embedding = embedding


class SequentialLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None):
        super(SequentialLink, self).__init__(input_size, output_size, z_size, depth, params, embedding)


# ============================================== LINK CLASSES ==========================================================

class MLPLink(BaseLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None):
        super(MLPLink, self).__init__(input_size, output_size, z_size, depth, params, embedding)

        self.mlp = nn.ModuleList([nn.Linear(input_size, output_size)] +
                                 [nn.Linear(input_size + output_size*i, output_size) for i in range(1, depth)])
        if embedding is not None:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size * depth, embedding.weight.shape[1])
                                                     for param in params})
        else:
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size * depth, z_size) for param in params})

    def forward(self, x, z_prev=None):
        outputs = []
        input = x
        for layer in self.mlp:
            outputs.append(layer(F.tanh(input)))# if len(outputs) else x)
            input = torch.cat([input, outputs[-1]], dim=-1)

        outputs = torch.cat(outputs, dim=-1)
        z_params = {param: activation(self.hidden_to_z_params[param](outputs)) for param, activation in
                    self.params.items()}
        if self.embedding is not None:
            z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))

        return z_params


class GRULink(SequentialLink):
    def __init__(self, input_size, output_size, z_size, depth, params, embedding=None):
        super(GRULink, self).__init__(input_size, output_size, z_size, depth, params, embedding)
        self.rnn = nn.GRU(input_size, output_size, depth, batch_first=True)
        if embedding is not None:
            self.project_z_prev = nn.Linear(embedding.weight.shape[1], output_size * depth)
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size * depth, embedding.weight.shape[1])
                                                     for param in params})
        else:
            self.project_z_prev = nn.Linear(z_size, output_size*depth)
            self.hidden_to_z_params = nn.ModuleDict({param: nn.Linear(output_size*depth, z_size) for param in params})

    def forward(self, x, z_prev=None):

        h_prev = self.project_z_prev(z_prev).view(self.depth, x.shape[0], self.output_size) \
            if z_prev is not None else None
        rnn_out, h = self.rnn(x.unsqueeze(-2), hx=h_prev)
        # These parameters are those of q(zi|xi, z<i) for the current word i (even though it's called z_params and
        # not zi_params)
        reshaped_h = h.transpose(0, 1).reshape(x.shape[0], self.output_size * self.depth)
        z_params = {param: activation(self.hidden_to_z_params[param](reshaped_h))
                    for param, activation in self.params.items()}
        if self.embedding is not None:
            z_params['logits'] = torch.matmul(z_params['logits'], self.embedding.weight.transpose(0, 1))
        return z_params
