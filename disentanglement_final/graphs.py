# This file defines the links between variables in the inference and generation networks of the PoS_tagging task
from itertools import chain

import torch.nn as nn

from components.links import  ConditionalCoattentiveTransformerLink, LastStateMLPLink, \
    LSTMLink, ConditionalCoattentiveQKVTransformerLink, MLPLink
from components.links import CoattentiveTransformerLink2 as CoattentiveTransformerLink
from components.links import ConditionalCoattentiveBARTTransformerLink, CoattentiveBARTTransformerLink,\
    QKVBartTransformerLink, LVARTransformerLink, SQKVBartTransformerLink
from disentanglement_final.variables import *


# ============================== OLD GRAPHS ============================================================================
def get_structured_auto_regressive_indep_graph(h_params, word_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=(i == 0)) for i in range(n_lvls)]
    zs_xprev_to_x = ConditionalCoattentiveTransformerLink(xin_size, zout_size,
                                                          xout_size,h_params.decoder_l, Categorical.parameter_activations,
                                                          word_embeddings, highway=h_params.highway, sbn=None,
                                                          dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                          memory=[z.name for z in z_gens], targets=['x_prev'],
                                                          nheads=4, bidirectional=False,
                                                          mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                          minimal_enc=h_params.minimal_enc)
    z_prior = [CoattentiveTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1], h_params.decoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=None,
                                          memory=[z_gens[i].name],
                                          n_mems=h_params.n_latents[i], mem_size=int(z_sizes[i]/h_params.n_latents[i]),
                                          highway=h_params.highway, dropout=h_params.dropout,
                                          n_targets=h_params.n_latents[i+1])
               for i in range(n_lvls-1)]
    number_parameters  = sum(p.numel() for p in zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)]
    z_posterior = [CoattentiveTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations, nheads=4,
                                              sequence=['x'], memory=None,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              highway=h_params.highway, dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i]) for i in range(n_lvls)]
    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]
    # for retrocompatibility:
    infer_edges = [infer_edges[0]]+infer_edges[n_lvls:]+infer_edges[1:n_lvls]
    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)]+\
                [nn.ModuleList([var, zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]]


    return {'infer': nn.ModuleList(infer_edges),
            'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_vanilla_graph(h_params, word_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gen = ZGeni(h_params, None, 0, allow_prior=True)
    z_xprev_to_x = LSTMLink(xin_size+zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                            Categorical.parameter_activations, dropout=h_params.dropout)

    # Inference
    x_inf, z_inf = XInfer(h_params, word_embeddings, has_rep=False), ZInferi(h_params, None, 0)
    x_to_z = LSTMLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout, last_state=True, bidirectional=True)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_z, z_inf])]),
            'gen':   nn.ModuleList([nn.ModuleList([xprev_gen, z_xprev_to_x, x_gen]),
                                   nn.ModuleList([z_gen, z_xprev_to_x, x_gen])])}, None, x_gen


# ============================== QKV GRAPHS ============================================================================

def get_qkv_graph(h_params, word_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    # zstin_size, zstout_size = int(h_params.z_size/max(h_params.n_latents)), \
    #                           int(h_params.z_size/max(h_params.n_latents))
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=(i == 0)) for i in range(n_lvls)]
    zst_gen = ZStGen(h_params)
    zst_zs_xprev_to_x = ConditionalCoattentiveQKVTransformerLink(xin_size, zout_size, xout_size, h_params.decoder_l,
                                                                 Categorical.parameter_activations, word_embeddings,
                                                                 highway=h_params.highway, sbn=None,
                                                                 dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                                 memory=[z.name for z in z_gens], targets=['x_prev'],
                                                                 key=['zs'], nheads=4, bidirectional=False,
                                                                 mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                                 minimal_enc=h_params.minimal_enc, n_keys=n_keys)
    z_prior = [CoattentiveTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                          h_params.decoder_l, Gaussian.parameter_activations, nheads=4, sequence=None,
                                          memory=[z_gens[i].name], n_mems=h_params.n_latents[i],
                                          mem_size=int(z_sizes[i]/h_params.n_latents[i]), highway=h_params.highway,
                                          dropout=h_params.dropout, n_targets=h_params.n_latents[i+1])
               for i in range(n_lvls-1)]
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    if len(z_prior):
        print("prior net sizes:")
        for i in range(len(z_prior)):
            number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
            print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStInfer(h_params)
    z_posterior = [CoattentiveTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations, nheads=4,
                                              sequence=['x'], memory=None,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              highway=h_params.highway, dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i]) for i in range(n_lvls)]
    x_to_zst = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]
    number_parameters = sum(p.numel() for p in z_posterior[0].parameters() if p.requires_grad)
    print("x to z1 size:", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in x_to_zst.parameters() if p.requires_grad)
    print("x to zst size:", "{0:05.2f} M".format(number_parameters/1e6))

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_qkv_graph2(h_params, word_embeddings):
    # This a version of the network where the hparams to dimensions configuration is reworked
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    # zstin_size, zstout_size = int(h_params.z_size/max(h_params.n_latents)), \
    #                           int(h_params.z_size/max(h_params.n_latents))
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=(i == 0)) for i in range(n_lvls)]
    zst_gen = ZStGen(h_params)
    zst_zs_xprev_to_x = ConditionalCoattentiveQKVTransformerLink(xin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                                                 Categorical.parameter_activations, word_embeddings,
                                                                 highway=h_params.highway, sbn=None,
                                                                 dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                                 memory=[z.name for z in z_gens], targets=['x_prev'],
                                                                 key=['zs'], nheads=4, bidirectional=False,
                                                                 mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                                 minimal_enc=h_params.minimal_enc, n_keys=n_keys)
    z_prior = [CoattentiveTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                          h_params.decoder_l, Gaussian.parameter_activations, nheads=4, sequence=None,
                                          memory=[z_gens[i].name], n_mems=h_params.n_latents[i],
                                          mem_size=int(z_sizes[i]/h_params.n_latents[i]), highway=h_params.highway,
                                          dropout=h_params.dropout, n_targets=h_params.n_latents[i+1])
               for i in range(n_lvls-1)]
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    if len(z_prior):
        print("prior net sizes:")
        for i in range(len(z_prior)):
            number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
            print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStInfer(h_params)
    z_posterior = [CoattentiveTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations, nheads=4,
                                              sequence=['x'], memory=None,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              highway=h_params.highway, dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i]) for i in range(n_lvls)]
    x_to_zst = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]
    number_parameters = sum(p.numel() for p in z_posterior[0].parameters() if p.requires_grad)
    print("x to z1 size:", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in x_to_zst.parameters() if p.requires_grad)
    print("x to zst size:", "{0:05.2f} M".format(number_parameters/1e6))

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_hqkv_graph(h_params, word_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    # zstin_size, zstout_size = int(h_params.z_size/max(h_params.n_latents)), \
    #                           int(h_params.z_size/max(h_params.n_latents))
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=False) for i in range(n_lvls)]
    zst_gen, zg_gen = ZStGen(h_params, allow_prior=False), ZGGen(h_params, allow_prior=True)
    zst_zs_xprev_to_x = ConditionalCoattentiveQKVTransformerLink(xin_size, zout_size, xout_size, h_params.decoder_l,
                                                                 Categorical.parameter_activations, word_embeddings,
                                                                 highway=h_params.highway, sbn=None,
                                                                 dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                                 memory=[z.name for z in z_gens], targets=['x_prev'],
                                                                 key=['zs'], nheads=4, bidirectional=False,
                                                                 mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                                 minimal_enc=h_params.minimal_enc, n_keys=n_keys)
    z_prior = [CoattentiveTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                          h_params.decoder_l, Gaussian.parameter_activations, nheads=4, sequence=None,
                                          memory=[z_gens[i].name], n_mems=h_params.n_latents[i],
                                          mem_size=int(z_sizes[i]/h_params.n_latents[i]), highway=h_params.highway,
                                          dropout=h_params.dropout, n_targets=h_params.n_latents[i+1])
               for i in range(n_lvls-1)]

    zg_to_z = MLPLink(zstin_size, h_params.decoder_h, z_sizes[0], h_params.decoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout)
    zg_to_zst = MLPLink(zstin_size, h_params.decoder_h, zstout_size, h_params.decoder_l, Gaussian.parameter_activations,
                        dropout=h_params.dropout)
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStInfer(h_params)
    zg_inf = ZGInfer(h_params)
    z_posterior = [CoattentiveTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations, nheads=4,
                                              sequence=['x'], memory=None,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              highway=h_params.highway, dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i]) for i in range(n_lvls)]
    x_to_zst = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    x_to_zg = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]+[nn.ModuleList([x_inf, x_to_zg, zg_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]+ \
                [nn.ModuleList([zg_gen, zg_to_z, z_gens[0]])]+[nn.ModuleList([zg_gen, zg_to_zst, zst_gen])]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_hqkv_graph_discrete_zs(h_params, word_embeddings):
    zs_emb = nn.Embedding(h_params.z_size, h_params.z_size)
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=False) for i in range(n_lvls)]
    zst_gen, zg_gen = ZStDiscGen(h_params, zs_emb, allow_prior=False), ZGGen(h_params, allow_prior=True)
    zst_zs_xprev_to_x = ConditionalCoattentiveQKVTransformerLink(xin_size, zout_size, xout_size, h_params.decoder_l,
                                                                 Categorical.parameter_activations, word_embeddings,
                                                                 highway=h_params.highway, sbn=None,
                                                                 dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                                 memory=[z.name for z in z_gens], targets=['x_prev'],
                                                                 key=['zs'], nheads=4, bidirectional=False,
                                                                 mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                                 minimal_enc=h_params.minimal_enc, n_keys=n_keys)
    z_prior = [CoattentiveTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                          h_params.decoder_l, Gaussian.parameter_activations, nheads=4, sequence=None,
                                          memory=[z_gens[i].name], n_mems=h_params.n_latents[i],
                                          mem_size=int(z_sizes[i]/h_params.n_latents[i]), highway=h_params.highway,
                                          dropout=h_params.dropout, n_targets=h_params.n_latents[i+1])
               for i in range(n_lvls-1)]

    zg_to_z = MLPLink(zstin_size, h_params.decoder_h, z_sizes[0], h_params.decoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout)
    zg_to_zst = MLPLink(zstin_size, h_params.decoder_h, zstout_size, h_params.decoder_l,
                        Categorical.parameter_activations, dropout=h_params.dropout)
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStDiscInfer(h_params, zs_emb)
    zg_inf = ZGInfer(h_params)
    z_posterior = [CoattentiveTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations, nheads=4,
                                              sequence=['x'], memory=None,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              highway=h_params.highway, dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i]) for i in range(n_lvls)]
    x_to_zst = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Categorical.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    x_to_zg = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]+[nn.ModuleList([x_inf, x_to_zg, zg_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]+ \
                [nn.ModuleList([zg_gen, zg_to_z, z_gens[0]])]+[nn.ModuleList([zg_gen, zg_to_zst, zst_gen])]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_hqkv_graph_old(h_params, word_embeddings):
    from components.links import CoattentiveTransformerLink
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    zstin_size, zstout_size = int(h_params.z_size/max(h_params.n_latents)), \
                              int(h_params.z_size/max(h_params.n_latents))
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=False) for i in range(n_lvls)]
    zst_gen, zg_gen = ZStGenLeg(h_params, allow_prior=False), ZGGenLeg(h_params, allow_prior=True)
    zst_zs_xprev_to_x = ConditionalCoattentiveQKVTransformerLink(xin_size, zout_size, xout_size, h_params.decoder_l,
                                                                 Categorical.parameter_activations, word_embeddings,
                                                                 highway=h_params.highway, sbn=None,
                                                                 dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                                 memory=[z.name for z in z_gens], targets=['x_prev'],
                                                                 key=['zs'], nheads=4, bidirectional=False,
                                                                 mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                                 minimal_enc=h_params.minimal_enc, n_keys=n_keys,
                                                                 old_ver=True, simple_zs_use=False)
    z_prior = [CoattentiveTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                          h_params.decoder_l, Gaussian.parameter_activations, nheads=4, sequence=None,
                                          memory=[z_gens[i].name], n_mems=h_params.n_latents[i],
                                          mem_size=int(z_sizes[i]/h_params.n_latents[i]), highway=h_params.highway,
                                          dropout=h_params.dropout, n_targets=h_params.n_latents[i+1])
               for i in range(n_lvls-1)]

    zg_to_z = MLPLink(zstin_size, h_params.decoder_h, z_sizes[0], h_params.decoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout)
    zg_to_zst = MLPLink(zstin_size, h_params.decoder_h, zstout_size, h_params.decoder_l, Gaussian.parameter_activations,
                        dropout=h_params.dropout)
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStInferLeg(h_params)
    zg_inf = ZGInferLeg(h_params)
    z_posterior = [CoattentiveTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations, nheads=4,
                                              sequence=['x'], memory=None,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              highway=h_params.highway, dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i]) for i in range(n_lvls)]
    x_to_zst = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    x_to_zg = CoattentiveTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, nheads=4, sequence=['x'],
                                          dropout=h_params.dropout, n_targets=1)
    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]+[nn.ModuleList([x_inf, x_to_zg, zg_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]+ \
                [nn.ModuleList([zg_gen, zg_to_z, z_gens[0]])]+[nn.ModuleList([zg_gen, zg_to_zst, zst_gen])]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


# ============================ QKV BART GRAPHS =========================================================================

def get_BARTADVAE(h_params, word_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=(i == 0)) for i in range(n_lvls)]
    zs_xprev_to_x = ConditionalCoattentiveBARTTransformerLink(xin_size, h_params.decoder_h, xout_size,h_params.decoder_l,
                                                              Categorical.parameter_activations, word_embeddings,
                                                              highway=h_params.highway, sbn=None,
                                                              dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                              memory=[z.name for z in z_gens], targets=['x_prev'],
                                                              bidirectional=False,
                                                              mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                              fr=h_params.fr)
    z_prior = [ConditionalCoattentiveBARTTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]),
                                                         z_sizes[i+1], h_params.decoder_l,
                                          Gaussian.parameter_activations,
                                          memory=[z_gens[i].name],
                                          n_mems=h_params.n_latents[i], mem_size=int(z_sizes[i]/h_params.n_latents[i]),
                                          highway=h_params.highway, dropout=h_params.dropout, fr=h_params.fr)
               for i in range(n_lvls-1)]
    # Sharing BART layers
    for p in z_prior:
        p.transformer = zs_xprev_to_x.transformer

    number_parameters  = sum(p.numel() for p in zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)]
    z_posterior = [CoattentiveBARTTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                                  z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations,
                                                  n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                                  dropout=h_params.dropout,
                                                  n_targets=h_params.n_latents[i],
                                                  fr=h_params.fr) for i in range(n_lvls)]
    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]
    # for retrocompatibility:
    infer_edges = [infer_edges[0]]+infer_edges[n_lvls:]+infer_edges[1:n_lvls]
    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)]+\
                [nn.ModuleList([var, zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_qkv_graphBART(h_params, word_embeddings):
    # This a version of the network where the hparams to dimensions configuration is reworked
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    # zstin_size, zstout_size = int(h_params.z_size/max(h_params.n_latents)), \
    #                           int(h_params.z_size/max(h_params.n_latents))
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=(i == 0)) for i in range(n_lvls)]
    zst_gen = ZStGen(h_params)

    zst_zs_xprev_to_x = QKVBartTransformerLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                             Categorical.parameter_activations, word_embeddings,
                                             highway=h_params.highway, sbn=None,
                                             dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                             memory=[z.name for z in z_gens], targets=['x_prev'],
                                             key=['zs'], nheads=4, bidirectional=False, mem_enc=h_params.tr_enc_in_dec,
                                             mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                             minimal_enc=h_params.minimal_enc, n_keys=n_keys,
                                               layer_wise=h_params.layer_wise_qkv, fr=h_params.fr)
    z_prior = [CoattentiveBARTTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                              h_params.decoder_l, Gaussian.parameter_activations,
                                              n_mems=h_params.n_latents[i], dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i+1], fr=h_params.fr)
               for i in range(n_lvls-1)]
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    if len(z_prior):
        print("prior net sizes:")
        for i in range(len(z_prior)):
            number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
            print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStInfer(h_params)
    z_posterior = [CoattentiveBARTTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                                  z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations,
                                                  n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                                  dropout=h_params.dropout,
                                                  n_targets=h_params.n_latents[i], fr=h_params.fr) for i in range(n_lvls)]
    x_to_zst = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, dropout=h_params.dropout, n_targets=1, fr=h_params.fr)
    # Sharing BART encoder
    for link in z_posterior:
        link.transformer = x_to_zst.transformer

    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]
    number_parameters = sum(p.numel() for p in z_posterior[0].parameters() if p.requires_grad)
    print("x to z1 size:", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in x_to_zst.parameters() if p.requires_grad)
    print("x to zst size:", "{0:05.2f} M".format(number_parameters/1e6))

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_min_struct_qkv_graphBART(h_params, word_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    # zstin_size, zstout_size = int(h_params.z_size/max(h_params.n_latents)), \
    #                           int(h_params.z_size/max(h_params.n_latents))
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=True) for i in range(n_lvls)]
    zst_gen = ZStGen(h_params, allow_prior=False)
    zst_zs_xprev_to_x = QKVBartTransformerLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                               Categorical.parameter_activations, word_embeddings,
                                               highway=h_params.highway, sbn=None,
                                               dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                               memory=[z.name for z in z_gens], targets=['x_prev'],
                                               key=['zs'], nheads=4, bidirectional=False,
                                               mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                               minimal_enc=h_params.minimal_enc, n_keys=n_keys,
                                               layer_wise=h_params.layer_wise_qkv, fr=h_params.fr)
    z_prior = [CoattentiveBARTTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                              h_params.decoder_l, Gaussian.parameter_activations,
                                              n_mems=h_params.n_latents[i], dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i+1], fr=h_params.fr)
               for i in range(n_lvls-1)]

    z_to_zst = MLPLink(z_sizes[0], h_params.decoder_h, zstout_size, h_params.decoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout)
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStInfer(h_params)
    z_posterior = [CoattentiveBARTTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i], fr=h_params.fr) for i in range(n_lvls)]
    x_to_zst = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations,
                                          dropout=h_params.dropout, n_targets=1, fr=h_params.fr)
    # Sharing BART encoder
    for link in z_posterior:
        link.transformer = x_to_zst.transformer

    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]+ \
                [nn.ModuleList([z_gens[0], z_to_zst, zst_gen])]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen

def get_hqkv_graphBART(h_params, word_embeddings):
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    # zstin_size, zstout_size = int(h_params.z_size/max(h_params.n_latents)), \
    #                           int(h_params.z_size/max(h_params.n_latents))
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=False) for i in range(n_lvls)]
    zst_gen, zg_gen = ZStGen(h_params, allow_prior=False), ZGGen(h_params, allow_prior=True)
    zst_zs_xprev_to_x = QKVBartTransformerLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                               Categorical.parameter_activations, word_embeddings,
                                               highway=h_params.highway, sbn=None,
                                               dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                               memory=[z.name for z in z_gens], targets=['x_prev'],
                                               key=['zs'], nheads=4, bidirectional=False,
                                               mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                               minimal_enc=h_params.minimal_enc, n_keys=n_keys,
                                               layer_wise=h_params.layer_wise_qkv, fr=h_params.fr)
    z_prior = [CoattentiveBARTTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                              h_params.decoder_l, Gaussian.parameter_activations,
                                              n_mems=h_params.n_latents[i], dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i+1], fr=h_params.fr)
               for i in range(n_lvls-1)]

    zg_to_z = MLPLink(zstin_size, h_params.decoder_h, z_sizes[0], h_params.decoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout)
    zg_to_zst = MLPLink(zstin_size, h_params.decoder_h, zstout_size, h_params.decoder_l, Gaussian.parameter_activations,
                        dropout=h_params.dropout)
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStInfer(h_params)
    zg_inf = ZGInfer(h_params)
    z_posterior = [CoattentiveBARTTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i], fr=h_params.fr) for i in range(n_lvls)]
    x_to_zst = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations,
                                          dropout=h_params.dropout, n_targets=1, fr=h_params.fr)
    x_to_zg = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations,
                                          dropout=h_params.dropout, n_targets=1, fr=h_params.fr)
    # Sharing BART encoder
    for link in z_posterior:
        link.transformer = x_to_zst.transformer
    x_to_zg.transformer = x_to_zst.transformer

    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]+[nn.ModuleList([x_inf, x_to_zg, zg_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]+ \
                [nn.ModuleList([zg_gen, zg_to_z, z_gens[0]])]+[nn.ModuleList([zg_gen, zg_to_zst, zst_gen])]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen


def get_hqkv_graph_discrete_zsBART(h_params, word_embeddings):
    zs_emb = nn.Embedding(h_params.z_size, h_params.z_size)
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    z_repnet = None
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    z_sizes = [int(zin_size*lv_size_prop) for lv_size_prop in lv_size_props]
    # Generation
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gens = [ZGeni(h_params, z_repnet, i, allow_prior=False) for i in range(n_lvls)]
    zst_gen, zg_gen = ZStDiscGen(h_params, zs_emb, allow_prior=False), ZGGen(h_params, allow_prior=True)
    zst_zs_xprev_to_x = QKVBartTransformerLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                                                 Categorical.parameter_activations, word_embeddings,
                                                                 highway=h_params.highway, sbn=None,
                                                                 dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                                 memory=[z.name for z in z_gens], targets=['x_prev'],
                                                                 key=['zs'], nheads=4, bidirectional=False,
                                                                 mem_size=int(z_sizes[0]/h_params.n_latents[0]),
                                                                 minimal_enc=h_params.minimal_enc, n_keys=n_keys,
                                               layer_wise=h_params.layer_wise_qkv, fr=h_params.fr)
    z_prior = [CoattentiveBARTTransformerLink(z_sizes[i], int(h_params.decoder_h*lv_size_props[i+1]), z_sizes[i+1],
                                          h_params.decoder_l, Gaussian.parameter_activations, n_mems=h_params.n_latents[i],
                                          dropout=h_params.dropout, n_targets=h_params.n_latents[i+1], fr=h_params.fr)
               for i in range(n_lvls-1)]

    zg_to_z = MLPLink(zstin_size, h_params.decoder_h, z_sizes[0], h_params.decoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout)
    zg_to_zst = MLPLink(zstin_size, h_params.decoder_h, zstout_size, h_params.decoder_l,
                        Categorical.parameter_activations, dropout=h_params.dropout)
    number_parameters = sum(p.numel() for p in zst_zs_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))
    print("prior net sizes:")
    for i in range(len(z_prior)):
        number_parameters = sum(p.numel() for p in z_prior[i].parameters() if p.requires_grad)
        print("{0:05.2f} M".format(number_parameters/1e6))

    # Inference
    x_inf, z_infs, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), [ZInferi(h_params, z_repnet, i) for i in
                                                                       range(n_lvls)], ZStDiscInfer(h_params, zs_emb)
    zg_inf = ZGInfer(h_params)
    z_posterior = [CoattentiveBARTTransformerLink(xin_size, int(lv_size_props[i]*h_params.encoder_h),
                                              z_sizes[i], h_params.encoder_l, Gaussian.parameter_activations,
                                              n_mems=sum(h_params.n_latents[i+1:n_lvls]) or None,
                                              dropout=h_params.dropout,
                                              n_targets=h_params.n_latents[i], fr=h_params.fr) for i in range(n_lvls)]
    x_to_zst = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Categorical.parameter_activations,
                                          dropout=h_params.dropout, n_targets=1, fr=h_params.fr)
    x_to_zg = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations,
                                          dropout=h_params.dropout, n_targets=1, fr=h_params.fr)
    # Sharing BART encoder
    for link in z_posterior:
        link.transformer = x_to_zst.transformer
    x_to_zg.transformer = x_to_zst.transformer

    infer_edges = [nn.ModuleList([x_inf, z_posti, z_infi]) for z_posti, z_infi in zip(z_posterior, z_infs)]+\
                  [nn.ModuleList([x_inf, x_to_zst, zst_inf])]+[nn.ModuleList([x_inf, x_to_zg, zg_inf])]

    gen_edges = [nn.ModuleList([z_gens[i], z_prior[i], z_gens[i+1]]) for i in range(n_lvls-1)] +\
                [nn.ModuleList([var, zst_zs_xprev_to_x, x_gen]) for var in z_gens+[xprev_gen]+[zst_gen]]+ \
                [nn.ModuleList([zg_gen, zg_to_z, z_gens[0]])]+[nn.ModuleList([zg_gen, zg_to_zst, zst_gen])]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges)}, None, x_gen



# ============================ QKVNext BART GRAPHS =========================================================================

def get_qkvNext(h_params, word_embeddings):
    # This a version of the network where the hparams to dimensions configuration is reworked
    xin_size, zin_size = h_params.embedding_dim, h_params.z_size
    xout_size, zout_size = h_params.vocab_size, h_params.z_size
    n_keys = h_params.n_keys
    zstin_size, zstout_size = h_params.z_size, h_params.z_size
    zpin_size, zpout_size = h_params.z_size, h_params.z_size
    n_lvls = len(h_params.n_latents)
    lv_size_props = [lv_n/max(h_params.n_latents) for lv_n in h_params.n_latents]
    # =============================== Generation Graph ===============================
    x_gen, xprev_gen = XGen(h_params, word_embeddings), XPrevGen(h_params, word_embeddings, has_rep=False)
    z_gen = ZGeni(h_params, None, 0)
    z_prev_gen = ZprevGeni(h_params, None, 0)
    zp_gen = ZpGen(h_params)
    zst_gen = ZStGen(h_params, allow_prior=False)

    zst_zp_z_xprev_to_x = SQKVBartTransformerLink(zin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                                                 Categorical.parameter_activations, word_embeddings,
                                                 highway=h_params.highway, sbn=None,
                                                 dropout=h_params.dropout, n_mems=sum(h_params.n_latents),
                                                 memory=['z1'], predicate=['zp'], targets=['x_prev'], key=['zs'],
                                                 p_size=zp_gen.size, key_size=zst_gen.size,
                                                 mem_size=int(zin_size/h_params.n_latents[0]),
                                                 minimal_enc=h_params.minimal_enc, n_keys=n_keys, bart_l=h_params.bart_l,
                                                 layer_wise=h_params.layer_wise_qkv, fr=h_params.fr)
    zp_to_zst = MLPLink(zpin_size, h_params.decoder_h, zstout_size, h_params.decoder_l, Gaussian.parameter_activations,
                      dropout=h_params.dropout)
    z_zprev_p_to_z0 = LVARTransformerLink(zpin_size, h_params.decoder_h, zout_size, h_params.decoder_l,
                                   Gaussian.parameter_activations, dropout=h_params.dropout,
                                   n_lv=sum(h_params.n_latents)+1, lv_size=int(zin_size/h_params.n_latents[0]), z0_size=zpin_size,
                                   z0=[zp_gen.name], zc=[z_prev_gen.name], nheads=h_params.n_heads)
    number_parameters = sum(p.numel() for p in zst_zp_z_xprev_to_x.parameters() if p.requires_grad)
    print("reconstruction net size:", "{0:05.2f} M".format(number_parameters/1e6))

    gen_edges = [nn.ModuleList([var, zst_zp_z_xprev_to_x, x_gen]) for var in [z_gen, xprev_gen, zst_gen, zp_gen]]+\
        [nn.ModuleList([z_prev_gen, z_zprev_p_to_z0, z_gen]), nn.ModuleList([zp_gen, z_zprev_p_to_z0, z_gen]),
         nn.ModuleList([zp_gen, zp_to_zst, zst_gen])]
    # =============================== Inference Graph ===============================
    x_inf, z_inf, zp_inf, zst_inf = XInfer(h_params, word_embeddings, has_rep=False), ZInferi(h_params, None, 0),\
                            ZpInfer(h_params), ZStInfer(h_params)
    x_to_z = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zin_size, h_params.encoder_l,
                                            Gaussian.parameter_activations, n_mems=None, dropout=h_params.dropout,
                                            n_targets=h_params.n_latents[0], fr=h_params.fr, bart_l=h_params.bart_l)
    x_to_zst = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zstout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, dropout=h_params.dropout, n_targets=1,
                                              fr=h_params.fr, bart_l=h_params.bart_l)
    x_to_zp = CoattentiveBARTTransformerLink(xin_size, h_params.encoder_h, zpout_size, h_params.encoder_l,
                                          Gaussian.parameter_activations, dropout=h_params.dropout, n_targets=1,
                                             fr=h_params.fr, bart_l=h_params.bart_l)
    # Sharing BART encoder
    x_to_z.transformer = x_to_zst.transformer
    x_to_zp.transformer = x_to_zst.transformer
    x_to_z.input_to_hidden = x_to_zst.input_to_hidden
    x_to_zp.input_to_hidden= x_to_zst.input_to_hidden

    infer_edges = [nn.ModuleList([x_inf, x_to_z, z_inf]), nn.ModuleList([x_inf, x_to_zst, zst_inf]),
                   nn.ModuleList([x_inf, x_to_zp, zp_inf])]

    number_parameters = sum(p.numel() for p in x_to_z.parameters() if p.requires_grad)
    print("x to z1 size:", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in x_to_zst.parameters() if p.requires_grad)
    print("x to zst size:", "{0:05.2f} M".format(number_parameters/1e6))
    # =============================== Auxiliart Semantics Graph ===============================
    zp_aux, z0_aux, g_prev, g_aux = ZpGen(h_params), ZGeni(h_params, None, index=0), GPrevAuxGen(h_params), \
                                    GAuxGen(h_params)
    zp_z0_gprev_to_g = ConditionalCoattentiveQKVTransformerLink(zin_size,h_params.decoder_h,zout_size,h_params.aux_l,
                                                                Gaussian.parameter_activations,dropout=h_params.dropout,
                                                                bidirectional=True,n_mems=h_params.n_latents[0],
                                                                mem_size=int(zin_size/h_params.n_latents[0]),
                                                                key_size=zp_aux.size, memory=[z0_aux.name], key=[zp_aux.name],
                                                                targets=[g_prev.name],nheads=h_params.n_heads)
    aux_edges = [nn.ModuleList([var, zp_z0_gprev_to_g, g_aux]) for var in [zp_aux, z0_aux, g_prev]]

    return {'infer': nn.ModuleList(infer_edges), 'gen':   nn.ModuleList(gen_edges),
            'aux':nn.ModuleList(aux_edges)}, None, x_gen
