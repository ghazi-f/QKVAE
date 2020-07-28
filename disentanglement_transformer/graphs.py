# This file defines the links between variables in the inference and generation networks of the PoS_tagging task

import torch.nn as nn

from components.links import CoattentiveTransformerLink, ConditionalCoattentiveTransformerLink, LastStateMLPLink
from disentanglement_transformer.variables import *


def get_disentanglement_graph(h_params, word_embeddings, pos_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    z_repnet = nn.LSTM(h_params.z_size, h_params.z_size, h_params.text_rep_l,
                       batch_first=True,
                       dropout=h_params.dropout)
    lv_number = h_params.n_latents

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings, has_rep=False)
    x_gen, z_gen = XGen(h_params, word_embeddings), ZGen(h_params, z_repnet, allow_prior=True )
    z_xprev_to_x = ConditionalCoattentiveTransformerLink(xin_size, zout_size, xout_size,
                                                         h_params.decoder_l, Categorical.parameter_activations,
                                                         word_embeddings, highway=h_params.highway, sbn=None,
                                                         dropout=h_params.dropout, n_mems=lv_number,
                                                         memory=['z'], targets=['x_prev'], nheads=4)

    # Inference
    x_inf, z_inf = XInfer(h_params, word_embeddings, has_rep=False), ZInfer(h_params, z_repnet)
    x_to_z = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l,
                                        Gaussian.parameter_activations, nheads=4,
                                        highway=h_params.highway, dropout=h_params.dropout, n_targets=lv_number)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    ]),
            'gen':   nn.ModuleList([
                                    nn.ModuleList([z_gen, z_xprev_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, z_xprev_to_x, x_gen]),
                                    ])}, None, x_gen


def get_non_auto_regressive_disentanglement_graph(h_params, word_embeddings, pos_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    z_repnet = nn.LSTM(h_params.z_size, h_params.z_size, h_params.text_rep_l,
                       batch_first=True,
                       dropout=h_params.dropout)
    lv_number = h_params.n_latents

    # Generation
    x_gen, z_gen, zlstm_gen = XGen(h_params, word_embeddings), ZGen(h_params, z_repnet, allow_prior=True), \
                   ZlstmGen(h_params, z_repnet, allow_prior=True)
    z_zlstm_to_x = ConditionalCoattentiveTransformerLink(xin_size, zout_size, xout_size,
                                                         h_params.decoder_l, Categorical.parameter_activations,
                                                         word_embeddings, highway=h_params.highway, sbn=None,
                                                         dropout=h_params.dropout, n_mems=lv_number,
                                                         memory=['z'], targets=['zlstm'], nheads=4, bidirectional=True)

    # Inference
    x_inf, z_inf, zlstm_inf = XInfer(h_params, word_embeddings, has_rep=True), ZInfer(h_params, z_repnet), \
                              ZlstmInfer(h_params, z_repnet)
    x_to_z = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l,
                                        Gaussian.parameter_activations, nheads=4,
                                        highway=h_params.highway, dropout=h_params.dropout, n_targets=lv_number)

    x_to_zlstm = LastStateMLPLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                     highway=h_params.highway, dropout=h_params.dropout)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    nn.ModuleList([x_inf, x_to_zlstm, zlstm_inf])
                                    ]),
            'gen':   nn.ModuleList([
                                    nn.ModuleList([z_gen, z_zlstm_to_x, x_gen]),
                                    nn.ModuleList([zlstm_gen, z_zlstm_to_x, x_gen]),
                                    ])}, None, x_gen
