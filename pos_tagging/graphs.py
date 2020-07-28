# This file defines the links between variables in the inference and generation networks of the PoS_tagging task

import torch.nn as nn

from components.links import MLPLink, TransformerLink
from pos_tagging.variables import *


def get_residual_reversed_graph_postag(h_params, word_embeddings, pos_embeddings):
    xin_size, yembin_size, yvalin_size, zin_size = h_params.text_rep_h, h_params.pos_embedding_dim, \
                                                   h_params.pos_embedding_dim, h_params.z_size
    xout_size, yembout_size, yvalout_size, zout_size = h_params.vocab_size, h_params.pos_embedding_dim,\
                                                       h_params.tag_size, h_params.z_size
    z_repnet = nn.LSTM(h_params.z_size, h_params.z_size, h_params.text_rep_l,
                       batch_first=True,
                       dropout=h_params.dropout)

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, yemb_gen, z_gen = XGen(h_params, word_embeddings), YEmbGen(h_params), ZGen(h_params, z_repnet)
    xprev_to_yemb = MLPLink(xin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameter_activations,
                         highway=h_params.highway, dropout=h_params.dropout)
    xprev_yemb_to_z = MLPLink(xin_size+yembin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameter_activations,
                              highway=h_params.highway, dropout=h_params.dropout)
    z_to_x = MLPLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                           Categorical.parameter_activations, word_embeddings, highway=h_params.highway, sbn=None,
                           dropout=h_params.dropout)

    # Inference
    yval_inf = YvalInfer(h_params, pos_embeddings)
    x_inf, yemb_inf, z_inf = XInfer(h_params, word_embeddings), YEmbInfer(h_params), ZInfer(h_params, z_repnet)
    x_to_z = MLPLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                                         highway=h_params.highway, dropout=h_params.dropout,)
                                         #residual={'conditions': ['x_prev', 'yemb'], 'link': xprev_yemb_to_z},)

    x_to_yemb = MLPLink(xin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameter_activations,
                                      highway=h_params.highway, dropout=h_params.dropout,)
                                      #residual={'conditions': ['x_prev'], 'link': xprev_to_yemb},)

    yemb_to_yval = MLPLink(yembin_size, h_params.pos_h, yvalout_size, h_params.pos_l, Categorical.parameter_activations,
                           embedding=pos_embeddings,
                           highway=h_params.highway, dropout=h_params.dropout)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_yemb, yemb_inf]),
                                    nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    nn.ModuleList([yemb_inf, yemb_to_yval, yval_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([x_prev_gen, xprev_to_yemb, yemb_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_yemb_to_z, z_gen]),
                                    nn.ModuleList([yemb_gen, xprev_yemb_to_z, z_gen]),
                                    nn.ModuleList([z_gen, z_to_x, x_gen]),
                                    #nn.ModuleList([x_prev_gen, xprev_z_to_x, x_gen])
                                    ])}, yval_inf, x_gen


def get_lm_graph(h_params, word_embeddings, pos_embeddings):
    xin_size, yembin_size, yvalin_size, zin_size = h_params.text_rep_h, h_params.pos_embedding_dim, \
                                                   h_params.pos_embedding_dim, h_params.z_size
    xout_size, yembout_size, yvalout_size, zout_size = h_params.vocab_size, h_params.pos_embedding_dim, \
                                                       h_params.tag_size, h_params.z_size

    z_repnet = nn.LSTM(h_params.z_size, h_params.z_size, h_params.text_rep_l,
                       batch_first=True,
                       dropout=h_params.dropout)
    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, yemb_gen, z_gen = XGen(h_params, word_embeddings), YEmbGen(h_params), ZGen(h_params, z_repnet)

    xprev_to_z = MLPLink(xin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameter_activations,
                              highway=h_params.highway, dropout=h_params.dropout)
    z_to_x = MLPLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                         Categorical.parameter_activations, word_embeddings, highway=h_params.highway, sbn=None,
                         dropout=h_params.dropout)
    xprev_to_x = MLPLink(xin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                         Categorical.parameter_activations, word_embeddings, highway=h_params.highway, sbn=None,
                         dropout=h_params.dropout)

    # Inference

    return {'infer': nn.ModuleList([]),
            'gen':   nn.ModuleList([
                                    # nn.ModuleList([x_prev_gen, xprev_to_z, z_gen]),
                                    # nn.ModuleList([z_gen, z_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_to_x, x_gen]),
                                    ])}, None, x_gen


def get_residual_transformer_graph_postag(h_params, word_embeddings, pos_embeddings):
    xin_size, yembin_size, yvalin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, \
                                                   h_params.pos_embedding_dim, h_params.z_size
    xout_size, yembout_size, yvalout_size, zout_size = h_params.vocab_size, h_params.pos_embedding_dim,\
                                                       h_params.tag_size, h_params.z_size

    z_repnet = nn.LSTM(h_params.z_size, h_params.z_size, h_params.text_rep_l,
                       batch_first=True,
                       dropout=h_params.dropout)
    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings, has_rep=False)
    x_gen, yemb_gen, z_gen = XGen(h_params, word_embeddings), YEmbGen(h_params), ZGen(h_params, z_repnet)
    xprev_to_yemb = TransformerLink(xin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameter_activations,
                         highway=h_params.highway, dropout=h_params.dropout)
    xprev_yemb_to_z = TransformerLink(xin_size+yembin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameter_activations,
                              highway=h_params.highway, dropout=h_params.dropout)
    z_to_x = TransformerLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                Categorical.parameter_activations, word_embeddings, highway=h_params.highway, sbn=None,
                                dropout=h_params.dropout)

    # Inference
    yval_inf = YvalInfer(h_params, pos_embeddings)
    x_inf, yemb_inf, z_inf = XInfer(h_params, word_embeddings, has_rep=False), YEmbInfer(h_params), ZInfer(h_params, z_repnet)
    x_to_z = TransformerLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                                         highway=h_params.highway, dropout=h_params.dropout,
                                         #residual={'conditions': ['x_prev', 'yemb'], 'link': xprev_yemb_to_z},
                                         bidirectional=True)
    x_to_yemb = TransformerLink(xin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameter_activations,
                                      highway=h_params.highway, dropout=h_params.dropout,
                                      #residual={'conditions': ['x_prev'], 'link': xprev_to_yemb},
                                      bidirectional=True)
    yemb_to_yval = MLPLink(yembin_size, h_params.pos_h, yvalout_size, h_params.pos_l, Categorical.parameter_activations,
                           embedding=pos_embeddings,
                           highway=h_params.highway, dropout=h_params.dropout)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_yemb, yemb_inf]),
                                    #nn.ModuleList([x_prev_gen, x_xprev_to_yemb, yemb_inf]),
                                    nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    #nn.ModuleList([x_prev_gen, x_x_prev_yemb_to_z, z_inf]),
                                    #nn.ModuleList([yemb_inf, x_x_prev_yemb_to_z, z_inf]),
                                    nn.ModuleList([yemb_inf, yemb_to_yval, yval_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([x_prev_gen, xprev_to_yemb, yemb_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_yemb_to_z, z_gen]),
                                    nn.ModuleList([yemb_gen, xprev_yemb_to_z, z_gen]),
                                    nn.ModuleList([z_gen, z_to_x, x_gen]),
                                    ])}, yval_inf, x_gen

