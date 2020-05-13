# This file defines the links between variables in the inference and generation networks of the PoS_tagging task

import torch.nn as nn

from components.links import GRULink, MLPLink
from pos_tagging.variables import *

from components.latent_variables import SoftmaxBottleneck


def get_graph(h_params, word_embeddings, pos_embeddings):
    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size

    # Inference
    x_inf, y_inf, z_inf = XInfer(h_params, word_embeddings), YInfer(h_params, pos_embeddings), ZInfer(h_params)
    x_to_z = GRULink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameters)
    x_z_to_y = GRULink(xin_size+zin_size, h_params.encoder_h, yout_size, h_params.encoder_l, Categorical.parameters,
                       pos_embeddings)

    # Generation
    x_gen, y_gen, z_gen = XGen(h_params, word_embeddings), YGen(h_params, pos_embeddings), ZGen(h_params)
    z_to_y = GRULink(zin_size, h_params.decoder_h, yout_size, h_params.decoder_l, Categorical.parameters,
                     pos_embeddings)
    z_y_to_x = GRULink(yin_size+zin_size, h_params.decoder_h, xout_size, h_params.decoder_l, Categorical.parameters,
                       word_embeddings)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    nn.ModuleList([x_inf, x_z_to_y, y_inf]),
                                    nn.ModuleList([z_inf, x_z_to_y, y_inf])]),
            'gen':   nn.ModuleList([nn.ModuleList([z_gen, z_to_y, y_gen]),
                                    nn.ModuleList([z_gen, z_y_to_x, x_gen]),
                                    nn.ModuleList([y_gen, z_y_to_x, x_gen])])}, y_inf, x_gen



def get_graph_minimal(h_params, word_embeddings, pos_embeddings):
    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size

    # Inference
    x_inf, y_inf, z_inf = XInfer(h_params, word_embeddings), YInfer(h_params, pos_embeddings), ZInfer(h_params)
    x_to_z = MLPLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameters)

    # Generation
    x_gen, y_gen, z_gen = XGen(h_params, word_embeddings), YGen(h_params, pos_embeddings), ZGen(h_params)
    z_to_x = MLPLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l, Categorical.parameters,
                     word_embeddings)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf])]),
            'gen':   nn.ModuleList([nn.ModuleList([z_gen, z_to_x, x_gen])])}, y_inf, x_gen


def get_graph_minimal_sequencial(h_params, word_embeddings, pos_embeddings):
    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size

    # Inference
    x_inf, y_inf, z_inf = XInfer(h_params, word_embeddings), YInfer(h_params, pos_embeddings), ZInfer(h_params)
    x_to_z = GRULink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameters)

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, y_gen, z_gen = XGen(h_params, word_embeddings), YGen(h_params, pos_embeddings), ZGen(h_params)
    z_xprev_to_x = MLPLink(zin_size+xin_size, h_params.decoder_h, xout_size, h_params.decoder_l, Categorical.parameters,
                           word_embeddings)
    xprev_to_z = GRULink(xin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameters)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf])]),
            'gen':   nn.ModuleList([nn.ModuleList([z_gen, z_xprev_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_to_z, z_gen]),
                                    nn.ModuleList([x_prev_gen, z_xprev_to_x, x_gen])])}, y_inf, x_gen


def get_graph_vsl(h_params, word_embeddings, pos_embeddings):
    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size

    # Inference
    x_inf, y_inf, z_inf = XInfer(h_params, word_embeddings), YInfer(h_params, pos_embeddings), ZInfer(h_params)
    x_to_z = MLPLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameters,
                     highway=h_params.highway)

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, y_gen, z_gen = XGen(h_params, word_embeddings), YGen(h_params, pos_embeddings), ZGen(h_params)
    z_xprev_to_x = MLPLink(zin_size+xin_size, h_params.decoder_h, xout_size, h_params.decoder_l, Categorical.parameters,
                           word_embeddings, highway=h_params.highway, sbn=None)#SoftmaxBottleneck())
    xprev_to_z = GRULink(xin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameters,
                         highway=h_params.highway)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf])]),
            'gen':   nn.ModuleList([nn.ModuleList([z_gen, z_xprev_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_to_z, z_gen]),
                                    nn.ModuleList([x_prev_gen, z_xprev_to_x, x_gen])
                                    ])}, y_inf, x_gen


def get_graph_postag(h_params, word_embeddings, pos_embeddings):
    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size

    # Inference
    x_inf, y_inf, z_inf = XInfer(h_params, word_embeddings), YInfer(h_params, pos_embeddings), ZInfer(h_params)
    x_to_z = GRULink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameters,
                     highway=h_params.highway, dropout=h_params.dropout)
    x_to_y = GRULink(xin_size, h_params.pos_h, yout_size, h_params.pos_l, Categorical.parameters,
                     embedding=pos_embeddings, highway=h_params.highway, dropout=h_params.dropout)

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, y_gen, z_gen = XGen(h_params, word_embeddings), YGen(h_params, pos_embeddings), ZGen(h_params)
    xprev_to_z = GRULink(xin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameters,
                         highway=h_params.highway, dropout=h_params.dropout)
    xprev_z_to_y = GRULink(xin_size+zin_size, h_params.pos_h, yout_size, h_params.pos_l, Categorical.parameters,
                         embedding=pos_embeddings, highway=h_params.highway, dropout=h_params.dropout)
    z_xprev_y_to_x = MLPLink(zin_size+xin_size+yin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                             Categorical.parameters, word_embeddings, highway=h_params.highway, sbn=None,
                             dropout=h_params.dropout)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    nn.ModuleList([x_inf, x_to_y, y_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([x_prev_gen, xprev_to_z, z_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_z_to_y, y_gen]),
                                    nn.ModuleList([x_prev_gen, z_xprev_y_to_x, x_gen]),
                                    nn.ModuleList([z_gen, xprev_z_to_y, y_gen]),
                                    nn.ModuleList([z_gen, z_xprev_y_to_x, x_gen]),
                                    nn.ModuleList([y_gen, z_xprev_y_to_x, x_gen])
                                    ])}, y_inf, x_gen


