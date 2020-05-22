# This file defines the links between variables in the inference and generation networks of the PoS_tagging task

import torch.nn as nn

from components.links import GRULink, MLPLink
from pos_tagging.variables import *

from components.latent_variables import SoftmaxBottleneck


def get_graph_postag(h_params, word_embeddings, pos_embeddings):
    xin_size, yembin_size, yvalin_size, zin_size = h_params.text_rep_h, h_params.pos_embedding_dim, \
                                                   h_params.pos_embedding_dim, h_params.z_size
    xout_size, yembout_size, yvalout_size, zout_size = h_params.vocab_size, h_params.pos_embedding_dim,\
                                                       h_params.tag_size, h_params.z_size

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, yemb_gen, z_gen = XGen(h_params, word_embeddings), YEmbGen(h_params), ZGen(h_params)
    xprev_to_z = GRULink(xin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameters,
                         highway=h_params.highway, dropout=h_params.dropout)
    xprev_z_to_yemb = MLPLink(xin_size+zin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameters,
                              embedding=None,
                              highway=h_params.highway, dropout=h_params.dropout)
    z_xprev_yemb_to_x = GRULink(zin_size+xin_size+yembin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                Categorical.parameters, word_embeddings, highway=h_params.highway, sbn=None,
                                dropout=h_params.dropout)

    # Inference
    yval_inf = YvalInfer(h_params, pos_embeddings)
    x_inf, yemb_inf, z_inf = XInfer(h_params, word_embeddings), YEmbInfer(h_params), ZInfer(h_params)
    x_to_z = GRULink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameters,
                     highway=h_params.highway, dropout=h_params.dropout)
    x_to_yemb = GRULink(xin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameters,
                        embedding=None,
                        highway=h_params.highway, dropout=h_params.dropout)
    yemb_to_yval = MLPLink(yembin_size, h_params.pos_h, yvalout_size, h_params.pos_l, Categorical.parameters,
                           embedding=pos_embeddings,
                           highway=h_params.highway, dropout=h_params.dropout)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    nn.ModuleList([x_inf, x_to_yemb, yemb_inf]),
                                    nn.ModuleList([yemb_inf, yemb_to_yval, yval_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([x_prev_gen, xprev_to_z, z_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_z_to_yemb, yemb_gen]),
                                    nn.ModuleList([x_prev_gen, z_xprev_yemb_to_x, x_gen]),
                                    nn.ModuleList([z_gen, xprev_z_to_yemb, yemb_gen]),
                                    nn.ModuleList([z_gen, z_xprev_yemb_to_x, x_gen]),
                                    nn.ModuleList([yemb_gen, z_xprev_yemb_to_x, x_gen])
                                    ])}, yval_inf, x_gen



def get_residual_graph_postag(h_params, word_embeddings, pos_embeddings):
    xin_size, yembin_size, yvalin_size, zin_size = h_params.text_rep_h, h_params.pos_embedding_dim, \
                                                   h_params.pos_embedding_dim, h_params.z_size
    xout_size, yembout_size, yvalout_size, zout_size = h_params.vocab_size, h_params.pos_embedding_dim,\
                                                       h_params.tag_size, h_params.z_size

    # Generation
    x_prev_gen = XPrevGen(h_params, word_embeddings)
    x_gen, yemb_gen, z_gen = XGen(h_params, word_embeddings), YEmbGen(h_params), ZGen(h_params)
    xprev_to_z = MLPLink(xin_size, h_params.decoder_h, zout_size, h_params.decoder_l, Gaussian.parameters,
                         highway=h_params.highway, dropout=h_params.dropout)
    xprev_z_to_yemb = MLPLink(xin_size+zin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameters,
                              highway=h_params.highway, dropout=h_params.dropout)
    z_xprev_yemb_to_x = MLPLink(zin_size+xin_size+yembin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                Categorical.parameters, word_embeddings, highway=h_params.highway, sbn=None,
                                dropout=h_params.dropout)

    # Inference
    yval_inf = YvalInfer(h_params, pos_embeddings)
    x_inf, yemb_inf, z_inf = XInfer(h_params, word_embeddings), YEmbInfer(h_params), ZInfer(h_params)
    x_x_prev_to_z = MLPLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameters,
                            highway=h_params.highway, dropout=h_params.dropout, residual={'conditions': ['x_prev'],
                                                                                          'link': xprev_to_z})
    x_xprev_z_to_yemb = MLPLink(xin_size, h_params.pos_h, yembout_size, h_params.pos_l, Gaussian.parameters,
                                highway=h_params.highway, dropout=h_params.dropout, residual={'conditions': ['x_prev', 'z'],
                                                                                              'link': xprev_z_to_yemb})
    yemb_to_yval = MLPLink(yembin_size, h_params.pos_h, yvalout_size, h_params.pos_l, Categorical.parameters,
                           embedding=pos_embeddings,
                           highway=h_params.highway, dropout=h_params.dropout)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_x_prev_to_z, z_inf]),
                                    nn.ModuleList([x_prev_gen, x_x_prev_to_z, z_inf]),
                                    nn.ModuleList([x_inf, x_xprev_z_to_yemb, yemb_inf]),
                                    nn.ModuleList([x_prev_gen, x_xprev_z_to_yemb, yemb_inf]),
                                    nn.ModuleList([z_inf, x_xprev_z_to_yemb, yemb_inf]),
                                    nn.ModuleList([x_inf, x_xprev_z_to_yemb, yemb_inf]),
                                    nn.ModuleList([yemb_inf, yemb_to_yval, yval_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([x_prev_gen, xprev_to_z, z_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_z_to_yemb, yemb_gen]),
                                    nn.ModuleList([x_prev_gen, z_xprev_yemb_to_x, x_gen]),
                                    nn.ModuleList([z_gen, xprev_z_to_yemb, yemb_gen]),
                                    nn.ModuleList([z_gen, z_xprev_yemb_to_x, x_gen]),
                                    nn.ModuleList([yemb_gen, z_xprev_yemb_to_x, x_gen])
                                    ])}, yval_inf, x_gen


