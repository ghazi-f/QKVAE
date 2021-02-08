# This file defines the links between variables in the inference and generation networks of the PoS_tagging task

import torch.nn as nn

from components.links import MLPLink, LastStateMLPLink, LSTMLink, DANLink
from taln.variables import *


def get_generation_graph(h_params, word_embeddings):

    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, z_gen = XGen(h_params, word_embeddings), ZGen(h_params, allow_prior=True)
    xprev_z_to_x = LSTMLink(xin_size+zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                            Categorical.parameter_activations, word_embeddings if h_params.tied_embeddings else None,
                            dropout=h_params.dropout)

    # Inference
    x_inf, z_inf = XInfer(h_params, word_embeddings), ZInfer(h_params)

    x_to_z = LSTMLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout, bidirectional=True, last_state=True)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([z_gen, xprev_z_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_z_to_x, x_gen])
                                    ])}, x_gen

