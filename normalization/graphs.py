# This file defines the links between variables in the inference and generation networks of the normalization task
import torch.nn as nn

from components.links import SublevelLSTMLink, LastStateMLPLink
from normalization.variables import *

def get_structured_auto_regressive_graph2(h_params, word_embeddings):
    # in this one, generation only depends on z3
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    z_repnet = None
    #nn.LSTM(h_params.z_size, h_params.z_size, h_params.text_rep_l,
    # batch_first=True,
    # dropout=h_params.dropout)
    lv_n1, lv_n2, lv_n3 = h_params.n_latents
    lv_size_prop1, lv_size_prop2, lv_size_prop3 = lv_n1/max(h_params.n_latents), lv_n2/max(h_params.n_latents),\
                                                     lv_n3/max(h_params.n_latents)
    z1_size, z2_size, z3_size = int(zin_size*lv_size_prop1), int(zin_size*lv_size_prop2), int(zin_size*lv_size_prop3)
    # Generation
    x_gen, z_gen1, z_gen2, z_gen3, xprev_gen = \
        XGen(h_params, word_embeddings), ZGen1(h_params, z_repnet, allow_prior=True), \
        ZGen2(h_params, z_repnet, allow_prior=True), ZGen3(h_params, z_repnet, allow_prior=True), \
        XPrevGen(h_params, word_embeddings, has_rep=False)
    z3_xprev_to_x = ConditionalCoattentiveTransformerLink(xin_size,
                                                                int(zout_size*lv_n3/max(h_params.n_latents)),
                                                                xout_size,h_params.decoder_l, Categorical.parameter_activations,
                                                                word_embeddings, highway=h_params.highway, sbn=None,
                                                                dropout=h_params.dropout, n_mems=lv_n3,
                                                                memory=['z3'], targets=['x_prev'],
                                                                nheads=4, bidirectional=False)
    z1_to_z2 = CoattentiveTransformerLink(z1_size, int(h_params.decoder_h*lv_size_prop2), z2_size, h_params.decoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=None, memory=['z1'],
                                          n_mems=h_params.n_latents[0],
                                          highway=h_params.highway, dropout=h_params.dropout, n_targets=lv_n2)
    z2_to_z3 = CoattentiveTransformerLink(z2_size, # Input size doesn't matter here because there's no input
                                             int(h_params.decoder_h*lv_size_prop3), z3_size, h_params.decoder_l,
                                             Gaussian.parameter_activations, nheads=4, sequence=None, memory=['z2'],
                                             n_mems=h_params.n_latents[1],
                                             highway=h_params.highway, dropout=h_params.dropout, n_targets=lv_n3)

    # Inference
    x_inf, z_inf1, z_inf2, z_inf3 = XInfer(h_params, word_embeddings, has_rep=False),\
                                          ZInfer1(h_params, z_repnet), \
                                          ZInfer2(h_params, z_repnet), ZInfer3(h_params, z_repnet)

    x_to_z3 = CoattentiveTransformerLink(xin_size, h_params.encoder_h, z3_size, h_params.encoder_l,
                                         Gaussian.parameter_activations, nheads=4, sequence=['x'], memory=None,
                                         highway=h_params.highway, dropout=h_params.dropout, n_targets=lv_n3)

    x_z3_to_z2 = CoattentiveTransformerLink(xin_size, int(lv_size_prop2*h_params.encoder_h), z2_size, h_params.encoder_l,
                                            Gaussian.parameter_activations, nheads=4, sequence=['x'], memory=['z3'],
                                            n_mems=lv_n3,
                                            highway=h_params.highway, dropout=h_params.dropout, n_targets=lv_n2)
    x_z2_z3_to_z1 = CoattentiveTransformerLink(xin_size, int(lv_size_prop1*h_params.encoder_h), z1_size, h_params.encoder_l,
                                        Gaussian.parameter_activations, nheads=4, sequence=['x'], memory=['z2', 'z3'],
                                        n_mems=lv_n3+lv_n2,
                                        highway=h_params.highway, dropout=h_params.dropout, n_targets=lv_n1)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_z2_z3_to_z1, z_inf1]),
                                    nn.ModuleList([z_inf2, x_z2_z3_to_z1, z_inf1]),
                                    nn.ModuleList([z_inf3, x_z2_z3_to_z1, z_inf1]),
                                    nn.ModuleList([z_inf3, x_z3_to_z2, z_inf2]),
                                    nn.ModuleList([x_inf, x_z3_to_z2, z_inf2]),
                                    nn.ModuleList([x_inf, x_to_z3, z_inf3]),
                                    ]),
            'gen':   nn.ModuleList([
                                    nn.ModuleList([z_gen1, z1_to_z2, z_gen2]),
                                    nn.ModuleList([z_gen2, z2_to_z3, z_gen3]),
                                    nn.ModuleList([z_gen3, z3_xprev_to_x, x_gen]),
                                    nn.ModuleList([xprev_gen, z3_xprev_to_x, x_gen]),
                                    ])}, None, x_gen


def get_normalization_graphs(h_params, c_embeddings, w_embeddings, y_embeddings):
    # Variable sizes
    cin_size, win_size, widin_size, zdiffin_size, \
    zcomin_size, yorigin_size = h_params.c_emb_size, h_params.w_emb_size, h_params.w_emb_size, h_params.zdiff_size,\
                                h_params.zcom_size, h_params.y_emb_size
    cout_size, wout_size, widout_size, zdiffout_size,\
    zcomout_size, yorigout_size = h_params.c_vocab_size, h_params.w_emb_size, h_params.w_vocab_size, \
                                  h_params.zdiff_size, h_params.zcom_size, 2
    # Noise variables
    nc_inf, nw_inf, nwid_inf, nzdiff_inf, nzcom_inf, nyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings),ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params),YorigInfer(h_params, y_embeddings)

    nc_gen, nw_gen, nwid_gen, nzdiff_gen, nzcom_gen, nyorig_gen = CGen(h_params, c_embeddings), WGen(h_params), \
                                                                  WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
                                                                  ZcomGen(h_params), YorigGen(h_params, y_embeddings)

    # Clean variables
    cc_inf, cw_inf, cwid_inf, czdiff_inf, czcom_inf, cyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings),ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params),YorigInfer(h_params, y_embeddings)

    cc_gen, cw_gen, cwid_gen, czdiff_gen, czcom_gen, cyorig_gen = CGen(h_params, c_embeddings), WGen(h_params), \
                                                                  WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
                                                                  ZcomGen(h_params), YorigGen(h_params, y_embeddings)
    # Inference network
    # Generation network

    return {'noise': {'infer': nn.ModuleList([]),
                      'gen':   nn.ModuleList([])},
            'clean': {'infer': nn.ModuleList([]),
                      'gen':   nn.ModuleList([])}}, None, x_gen

