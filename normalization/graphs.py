# This file defines the links between variables in the inference and generation networks of the normalization task
import torch.nn as nn

from components.links import SublevelLSTMLink, LastStateMLPLink, LSTMLink, MLPLink
from normalization.variables import *


def get_normalization_graphs(h_params, c_embeddings, w_embeddings, y_embeddings):
    # Variable sizes
    cin_size, win_size, widin_size, zdiffin_size, \
    zcomin_size, yorigin_size = h_params.c_embedding_dim, h_params.w_embedding_dim, h_params.w_embedding_dim,\
                                h_params.zdiff_size,h_params.zcom_size, h_params.y_embedding_dim
    cout_size, wout_size, widout_size, zdiffout_size,\
    zcomout_size, yorigout_size = h_params.c_vocab_size, h_params.w_embedding_dim, h_params.w_vocab_size, \
                                  h_params.zdiff_size, h_params.zcom_size, 2
    # Noise variables
    nc_inf, nw_inf, nwid_inf, nzdiff_inf, nzcom_inf, nyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    nc_gen, nw_gen, nwid_gen, nzdiff_gen, nzcom_gen, nyorig_gen, nwprev, n_cprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)

    # Clean variables
    cc_inf, cw_inf, cwid_inf, czdiff_inf, czcom_inf, cyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    cc_gen, cw_gen, cwid_gen, czdiff_gen, czcom_gen, cyorig_gen, cwprev, ccprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    # Inference network
    c_to_w = SublevelLSTMLink(cin_size, h_params.c_encoder_h, wout_size, h_params.c_encoder_l,
                              Gaussian.parameter_activations, dropout=h_params.dropout, bidirectional=True,
                              sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len, last_state=True)
    w_to_wid = LSTMLink(win_size, h_params.c_encoder_h, widout_size, h_params.c_encoder_l,
                        Categorical.parameter_activations, w_embeddings, dropout=h_params.dropout, bidirectional=True)
    w_to_zcom = LSTMLink(win_size, h_params.w_encoder_h, zcomout_size, h_params.w_encoder_l,
                         Gaussian.parameter_activations, dropout=h_params.dropout, last_state=True, bidirectional=True)
    w_to_zdiff = LSTMLink(win_size, h_params.w_encoder_h, zdiffout_size, h_params.w_encoder_l,
                         Gaussian.parameter_activations, dropout=h_params.dropout, last_state=True, bidirectional=True)
    zdiff_to_yorig = MLPLink(zdiffin_size, h_params.y_encoder_h, yorigout_size, h_params.y_encoder_l,
                             Categorical.parameter_activations, dropout=h_params.dropout)

    # Generation network
    yorig_to_zdiff = MLPLink(yorigin_size, h_params.y_encoder_h, zdiffout_size, h_params.y_encoder_l,
                             Gaussian.parameter_activations, dropout=h_params.dropout)
    zdiff_zcom_wprev_to_w = LSTMLink(zdiffin_size+zcomin_size+win_size, h_params.w_decoder_h, wout_size,
                                     h_params.w_decoder_l, Gaussian.parameter_activations, dropout=h_params.dropout)
    #TODO: revert to auto_regressive character generation if this one step char generation doesn't do it
    w_to_c = SublevelLSTMLink(win_size, h_params.c_decoder_h*h_params.c_max_len, cout_size, h_params.c_decoder_l,
                              Categorical.parameter_activations, embedding=c_embeddings, dropout=h_params.dropout,
                              bidirectional=True, sub_lvl_size=h_params.c_max_len, sub_lvl_vars=[])

    return {'noise': {'infer': nn.ModuleList([nn.ModuleList([nc_inf, c_to_w, nw_inf]),
                                              nn.ModuleList([nw_inf, w_to_wid, nwid_inf]),
                                              nn.ModuleList([nw_inf, w_to_zcom, nzcom_inf]),
                                              nn.ModuleList([nw_inf, w_to_zdiff, nzdiff_inf]),
                                              nn.ModuleList([nzdiff_inf, zdiff_to_yorig, nyorig_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([nyorig_gen, yorig_to_zdiff, nzdiff_gen]),
                                              nn.ModuleList([nzdiff_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nzcom_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nwprev, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nw_gen, w_to_c, nc_gen])])},
            'clean': {'infer': nn.ModuleList([nn.ModuleList([cc_inf, c_to_w, cw_inf]),
                                              nn.ModuleList([cw_inf, w_to_wid, cwid_inf]),
                                              nn.ModuleList([cw_inf, w_to_zcom, czcom_inf]),
                                              nn.ModuleList([cw_inf, w_to_zdiff, czdiff_inf]),
                                              nn.ModuleList([czdiff_inf, zdiff_to_yorig, cyorig_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([cyorig_gen, yorig_to_zdiff, czdiff_gen]),
                                              nn.ModuleList([czdiff_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([czcom_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cwprev, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cw_gen, w_to_c, cc_gen])])}}, \
           {'clean': cc_gen, 'noise': nc_gen},\
           {'clean': cwid_inf, 'noise': nwid_inf},\
           {'clean': nn.ModuleList([nn.ModuleList([cw_gen, w_to_wid, cwid_gen])]),
            'noise': nn.ModuleList([nn.ModuleList([nw_gen, w_to_wid, nwid_gen])])}


def get_reordered_normalization_graphs(h_params, c_embeddings, w_embeddings, y_embeddings):
    # Variable sizes
    cin_size, win_size, widin_size, zdiffin_size, \
    zcomin_size, yorigin_size = h_params.c_embedding_dim, h_params.w_embedding_dim, h_params.w_embedding_dim,\
                                h_params.zdiff_size,h_params.zcom_size, h_params.y_embedding_dim
    cout_size, wout_size, widout_size, zdiffout_size,\
    zcomout_size, yorigout_size = h_params.c_vocab_size, h_params.w_embedding_dim, h_params.w_vocab_size, \
                                  h_params.zdiff_size, h_params.zcom_size, 2
    # Noise variables
    nc_inf, nw_inf, nwid_inf, nzdiff_inf, nzcom_inf, nyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    nc_gen, nw_gen, nw_sup, nwid_gen, nzdiff_gen, nzcom_gen, nyorig_gen, nwprev, n_cprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)

    # Clean variables
    cc_inf, cw_inf, cwid_inf, czdiff_inf, czcom_inf, cyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    cc_gen, cw_gen, cw_sup, cwid_gen, czdiff_gen, czcom_gen, cyorig_gen, cwprev, ccprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    # Inference network
    c_to_yorig = SublevelLSTMLink(cin_size, h_params.y_encoder_h, yorigout_size, h_params.y_encoder_l,
                                  Categorical.parameter_activations, dropout=h_params.dropout, last_state=True,
                                  bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_yorig_to_zdiff = SublevelLSTMLink(cin_size+yorigin_size, h_params.zd_encoder_h, zdiffout_size,
                                        h_params.zd_encoder_l, Gaussian.parameter_activations, dropout=h_params.dropout,
                                        last_state=True, bidirectional=True, sub_lvl_vars=['c'],
                                        sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_to_zcom = SublevelLSTMLink(cin_size, h_params.zc_encoder_h, zcomout_size, h_params.zc_encoder_l,
                                 Gaussian.parameter_activations, dropout=h_params.dropout, last_state=True,
                                 bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_zcom_zdiff_to_w = SublevelLSTMLink(cin_size+zcomin_size+zdiffin_size, h_params.w_encoder_h, wout_size,
                                         h_params.w_encoder_l, Gaussian.parameter_activations, dropout=h_params.dropout,
                                         bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len,
                                         up_lvl=True)
    w_to_wid = MLPLink(win_size, h_params.w_encoder_h, widout_size, h_params.w_encoder_l,
                       Categorical.parameter_activations, w_embeddings, dropout=h_params.dropout)

    # Generation network
    yorig_to_zdiff = MLPLink(yorigin_size, h_params.zd_encoder_h, zdiffout_size, h_params.zd_encoder_l,
                             Gaussian.parameter_activations, dropout=h_params.dropout)
    zdiff_zcom_wprev_to_w = LSTMLink(zdiffin_size+zcomin_size+win_size, h_params.w_decoder_h, wout_size,
                                     h_params.w_decoder_l, Gaussian.parameter_activations, dropout=h_params.dropout)
    #TODO: revert to auto_regressive character generation if this one step char generation doesn't do it
    w_to_c = SublevelLSTMLink(win_size, h_params.c_decoder_h*h_params.c_max_len, cout_size, h_params.c_decoder_l,
                              Categorical.parameter_activations, #embedding=c_embeddings,
                              dropout=h_params.dropout,
                              bidirectional=True, sub_lvl_size=h_params.c_max_len, sub_lvl_vars=[])

    return {'noise': {'infer': nn.ModuleList([nn.ModuleList([nw_inf, w_to_wid, nwid_inf]),
                                              nn.ModuleList([nc_inf, c_to_yorig, nyorig_inf]),
                                              nn.ModuleList([nc_inf, c_yorig_to_zdiff, nzdiff_inf]),
                                              nn.ModuleList([nc_inf, c_to_zcom, nzcom_inf]),
                                              nn.ModuleList([nc_inf, c_zcom_zdiff_to_w, nw_inf]),
                                              nn.ModuleList([nyorig_inf, c_yorig_to_zdiff, nzdiff_inf]),
                                              nn.ModuleList([nzdiff_inf, c_zcom_zdiff_to_w, nw_inf]),
                                              nn.ModuleList([nzcom_inf, c_zcom_zdiff_to_w, nw_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([nyorig_gen, yorig_to_zdiff, nzdiff_gen]),
                                              nn.ModuleList([nzdiff_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nzcom_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nwprev, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nw_gen, w_to_c, nc_gen])])},
            'clean': {'infer': nn.ModuleList([nn.ModuleList([cw_inf, w_to_wid, cwid_inf]),
                                              nn.ModuleList([cc_inf, c_to_yorig, cyorig_inf]),
                                              nn.ModuleList([cc_inf, c_yorig_to_zdiff, czdiff_inf]),
                                              nn.ModuleList([cc_inf, c_to_zcom, czcom_inf]),
                                              nn.ModuleList([cc_inf, c_zcom_zdiff_to_w, cw_inf]),
                                              nn.ModuleList([cyorig_inf, c_yorig_to_zdiff, czdiff_inf]),
                                              nn.ModuleList([czdiff_inf, c_zcom_zdiff_to_w, cw_inf]),
                                              nn.ModuleList([czcom_inf, c_zcom_zdiff_to_w, cw_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([cyorig_gen, yorig_to_zdiff, czdiff_gen]),
                                              nn.ModuleList([czdiff_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([czcom_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cwprev, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cw_gen, w_to_c, cc_gen])])}}, \
           {'clean': cc_gen, 'noise': nc_gen},\
           {'clean': cwid_inf, 'noise': nwid_inf},\
           {'clean': nn.ModuleList([nn.ModuleList([cw_sup, w_to_wid, cwid_gen])]),
            'noise': nn.ModuleList([nn.ModuleList([nw_sup, w_to_wid, nwid_gen])])}


def get_reordered_indep_normalization_graphs(h_params, c_embeddings, w_embeddings, y_embeddings):
    # Variable sizes
    cin_size, win_size, widin_size, zdiffin_size, \
    zcomin_size, yorigin_size = h_params.c_embedding_dim, h_params.w_embedding_dim, h_params.w_embedding_dim,\
                                h_params.zdiff_size,h_params.zcom_size, h_params.y_embedding_dim
    cout_size, wout_size, widout_size, zdiffout_size,\
    zcomout_size, yorigout_size = h_params.c_vocab_size, h_params.w_embedding_dim, h_params.w_vocab_size, \
                                  h_params.zdiff_size, h_params.zcom_size, 2
    # Noise variables
    nc_inf, nw_inf, nwid_inf, nzdiff_inf, nzcom_inf, nyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    nc_gen, nw_gen, nw_sup, nwid_gen, nzdiff_gen, nzcom_gen, nyorig_gen, nwprev, n_cprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    rnc_gen, rnw_gen, rnw_sup, rnwid_gen, rnzdiff_gen, rnzcom_gen, rnyorig_gen, rnwprev, rn_cprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)

    # Clean variables
    cc_inf, cw_inf, cwid_inf, czdiff_inf, czcom_inf, cyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    cc_gen, cw_gen, cw_sup, cwid_gen, czdiff_gen, czcom_gen, cyorig_gen, cwprev, ccprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    rcc_gen, rcw_gen, rcw_sup, rcwid_gen, rczdiff_gen, rczcom_gen, rcyorig_gen, rcwprev, rccprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    # Inference network
    c_to_yorig = SublevelLSTMLink(cin_size, h_params.y_encoder_h, yorigout_size, h_params.y_encoder_l,
                                  Categorical.parameter_activations, dropout=h_params.dropout, last_state=True,
                                  bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_to_zdiff = SublevelLSTMLink(cin_size, h_params.zd_encoder_h, zdiffout_size,
                                        h_params.zd_encoder_l, Gaussian.parameter_activations, dropout=h_params.dropout,
                                        last_state=True, bidirectional=True, sub_lvl_vars=['c'],
                                        sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_to_zcom = SublevelLSTMLink(cin_size, h_params.zc_encoder_h, zcomout_size, h_params.zc_encoder_l,
                                 Gaussian.parameter_activations, dropout=h_params.dropout, last_state=True,
                                 bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_to_w = SublevelLSTMLink(cin_size, h_params.w_encoder_h, wout_size,
                                         h_params.w_encoder_l, Gaussian.parameter_activations, dropout=h_params.dropout,
                                         bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len,
                                         up_lvl=True)
    w_to_wid = MLPLink(win_size, h_params.w_encoder_h, widout_size, h_params.w_encoder_l,
                       Categorical.parameter_activations, w_embeddings, dropout=h_params.dropout)

    # Generation network
    yorig_to_zdiff = MLPLink(yorigin_size, h_params.zd_encoder_h, zdiffout_size, h_params.zd_encoder_l,
                             Gaussian.parameter_activations, dropout=h_params.dropout)
    zdiff_zcom_wprev_to_w = LSTMLink(zdiffin_size+zcomin_size+win_size, h_params.w_decoder_h, wout_size,
                                     h_params.w_decoder_l, Gaussian.parameter_activations, dropout=h_params.dropout)
    #TODO: revert to auto_regressive character generation if this one step char generation doesn't do it
    w_to_c = SublevelLSTMLink(win_size, h_params.c_decoder_h*h_params.c_max_len, cout_size, h_params.c_decoder_l,
                              Categorical.parameter_activations, #embedding=c_embeddings,
                              dropout=h_params.dropout,
                              bidirectional=True, sub_lvl_size=h_params.c_max_len, sub_lvl_vars=[])

    return {'noise': {'infer': nn.ModuleList([nn.ModuleList([nw_inf, w_to_wid, nwid_inf]),
                                              nn.ModuleList([nc_inf, c_to_yorig, nyorig_inf]),
                                              nn.ModuleList([nc_inf, c_to_zdiff, nzdiff_inf]),
                                              nn.ModuleList([nc_inf, c_to_zcom, nzcom_inf]),
                                              nn.ModuleList([nc_inf, c_to_w, nw_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([nyorig_gen, yorig_to_zdiff, nzdiff_gen]),
                                              nn.ModuleList([nzdiff_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nzcom_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nwprev, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nw_gen, w_to_c, nc_gen])]),
                      'rec_gen':   nn.ModuleList([nn.ModuleList([rnyorig_gen, yorig_to_zdiff, rnzdiff_gen]),
                                              nn.ModuleList([rnzdiff_gen, zdiff_zcom_wprev_to_w, rnw_gen]),
                                              nn.ModuleList([rnzcom_gen, zdiff_zcom_wprev_to_w, rnw_gen]),
                                              nn.ModuleList([rnwprev, zdiff_zcom_wprev_to_w, rnw_gen]),
                                              nn.ModuleList([rnw_gen, w_to_c, rnc_gen])])},
            'clean': {'infer': nn.ModuleList([nn.ModuleList([cw_inf, w_to_wid, cwid_inf]),
                                              nn.ModuleList([cc_inf, c_to_yorig, cyorig_inf]),
                                              nn.ModuleList([cc_inf, c_to_zdiff, czdiff_inf]),
                                              nn.ModuleList([cc_inf, c_to_zcom, czcom_inf]),
                                              nn.ModuleList([cc_inf, c_to_w, cw_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([cyorig_gen, yorig_to_zdiff, czdiff_gen]),
                                              nn.ModuleList([czdiff_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([czcom_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cwprev, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cw_gen, w_to_c, cc_gen])]),
                      'rec_gen':   nn.ModuleList([nn.ModuleList([rcyorig_gen, yorig_to_zdiff, rczdiff_gen]),
                                              nn.ModuleList([rczdiff_gen, zdiff_zcom_wprev_to_w, rcw_gen]),
                                              nn.ModuleList([rczcom_gen, zdiff_zcom_wprev_to_w, rcw_gen]),
                                              nn.ModuleList([rcwprev, zdiff_zcom_wprev_to_w, rcw_gen]),
                                              nn.ModuleList([rcw_gen, w_to_c, rcc_gen])])}}, \
           {'clean': cc_gen, 'noise': nc_gen},\
           {'clean': cwid_inf, 'noise': nwid_inf},\
           {'clean': nn.ModuleList([nn.ModuleList([cw_sup, w_to_wid, cwid_gen])]),
            'noise': nn.ModuleList([nn.ModuleList([nw_sup, w_to_wid, nwid_gen])])}


def get_c_prev_normalization_graphs(h_params, c_embeddings, w_embeddings, y_embeddings):
    # Variable sizes
    cin_size, win_size, widin_size, zdiffin_size, \
    zcomin_size, yorigin_size = h_params.c_embedding_dim, h_params.w_embedding_dim, h_params.w_embedding_dim,\
                                h_params.zdiff_size,h_params.zcom_size, h_params.y_embedding_dim
    cout_size, wout_size, widout_size, zdiffout_size,\
    zcomout_size, yorigout_size = h_params.c_vocab_size, h_params.w_embedding_dim, h_params.w_vocab_size, \
                                  h_params.zdiff_size, h_params.zcom_size, 2
    # Noise variables
    nc_inf, nw_inf, nwid_inf, nzdiff_inf, nzcom_inf, nyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    nc_gen, nw_gen, nw_sup, nwid_gen, nzdiff_gen, nzcom_gen, nyorig_gen, nwprev, ncprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    rnc_gen, rnw_gen, rnw_sup, rnwid_gen, rnzdiff_gen, rnzcom_gen, rnyorig_gen, rnwprev, rn_cprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)

    # Clean variables
    cc_inf, cw_inf, cwid_inf, czdiff_inf, czcom_inf, cyorig_inf = CInfer(h_params, c_embeddings), WInfer(h_params),\
                                                              WidInfer(h_params, w_embeddings), ZdiffInfer(h_params),\
                                                              ZcomInfer(h_params), YorigInfer(h_params, y_embeddings)

    cc_gen, cw_gen, cw_sup, cwid_gen, czdiff_gen, czcom_gen, cyorig_gen, cwprev, ccprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    rcc_gen, rcw_gen, rcw_sup, rcwid_gen, rczdiff_gen, rczcom_gen, rcyorig_gen, rcwprev, rccprev = \
        CGen(h_params, c_embeddings), WGen(h_params), WGen(h_params), WidGen(h_params, w_embeddings), ZdiffGen(h_params), \
        ZcomGen(h_params), YorigGen(h_params, y_embeddings), WPrevGen(h_params), CPrevGen(h_params, c_embeddings)
    # Inference network
    c_to_yorig = SublevelLSTMLink(cin_size, h_params.y_encoder_h, yorigout_size, h_params.y_encoder_l,
                                  Categorical.parameter_activations, dropout=h_params.dropout, last_state=True,
                                  bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_to_zdiff = SublevelLSTMLink(cin_size, h_params.zd_encoder_h, zdiffout_size,
                                        h_params.zd_encoder_l, Gaussian.parameter_activations, dropout=h_params.dropout,
                                        last_state=True, bidirectional=True, sub_lvl_vars=['c'],
                                        sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_to_zcom = SublevelLSTMLink(cin_size, h_params.zc_encoder_h, zcomout_size, h_params.zc_encoder_l,
                                 Gaussian.parameter_activations, dropout=h_params.dropout, last_state=True,
                                 bidirectional=True, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len, up_lvl=True)
    c_to_w = SublevelLSTMLink(cin_size, h_params.w_encoder_h, wout_size,
                                         h_params.w_encoder_l, Gaussian.parameter_activations, dropout=h_params.dropout,
                                         bidirectional=False, sub_lvl_vars=['c'], sub_lvl_size=h_params.c_max_len,
                                         up_lvl=True)
    w_to_wid = MLPLink(win_size, h_params.w_encoder_h, widout_size, h_params.w_encoder_l,
                       Categorical.parameter_activations, w_embeddings, dropout=h_params.dropout)

    # Generation network
    yorig_to_zdiff = MLPLink(yorigin_size, h_params.zd_encoder_h, zdiffout_size, h_params.zd_encoder_l,
                             Gaussian.parameter_activations, dropout=h_params.dropout)
    zdiff_zcom_wprev_to_w = LSTMLink(zdiffin_size+zcomin_size+win_size, h_params.w_decoder_h, wout_size,
                                     h_params.w_decoder_l, Gaussian.parameter_activations, dropout=h_params.dropout)
    #TODO: revert to auto_regressive character generation if this one step char generation doesn't do it
    w_cprev_to_c = SublevelLSTMLink(win_size+cin_size, h_params.c_decoder_h*h_params.c_max_len, cout_size
                                    , h_params.c_decoder_l, Categorical.parameter_activations, #embedding=c_embeddings,
                                    dropout=h_params.dropout, bidirectional=False, sub_lvl_size=h_params.c_max_len,
                                    sub_lvl_vars=['c_prev'])

    return {'noise': {'infer': nn.ModuleList([nn.ModuleList([nw_inf, w_to_wid, nwid_inf]),
                                              nn.ModuleList([nc_inf, c_to_yorig, nyorig_inf]),
                                              nn.ModuleList([nc_inf, c_to_zdiff, nzdiff_inf]),
                                              nn.ModuleList([nc_inf, c_to_zcom, nzcom_inf]),
                                              nn.ModuleList([nc_inf, c_to_w, nw_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([nyorig_gen, yorig_to_zdiff, nzdiff_gen]),
                                              nn.ModuleList([nzdiff_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nzcom_gen, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nwprev, zdiff_zcom_wprev_to_w, nw_gen]),
                                              nn.ModuleList([nw_gen, w_cprev_to_c, nc_gen]),
                                              nn.ModuleList([ncprev, w_cprev_to_c, nc_gen])]),
                      'rec_gen':   nn.ModuleList([nn.ModuleList([rnyorig_gen, yorig_to_zdiff, rnzdiff_gen]),
                                              nn.ModuleList([rnzdiff_gen, zdiff_zcom_wprev_to_w, rnw_gen]),
                                              nn.ModuleList([rnzcom_gen, zdiff_zcom_wprev_to_w, rnw_gen]),
                                              nn.ModuleList([rnwprev, zdiff_zcom_wprev_to_w, rnw_gen]),
                                              nn.ModuleList([rnw_gen, w_cprev_to_c, rnc_gen]),
                                              nn.ModuleList([ncprev, w_cprev_to_c, rnc_gen])])},
            'clean': {'infer': nn.ModuleList([nn.ModuleList([cw_inf, w_to_wid, cwid_inf]),
                                              nn.ModuleList([cc_inf, c_to_yorig, cyorig_inf]),
                                              nn.ModuleList([cc_inf, c_to_zdiff, czdiff_inf]),
                                              nn.ModuleList([cc_inf, c_to_zcom, czcom_inf]),
                                              nn.ModuleList([cc_inf, c_to_w, cw_inf])]),
                      'gen':   nn.ModuleList([nn.ModuleList([cyorig_gen, yorig_to_zdiff, czdiff_gen]),
                                              nn.ModuleList([czdiff_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([czcom_gen, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cwprev, zdiff_zcom_wprev_to_w, cw_gen]),
                                              nn.ModuleList([cw_gen, w_cprev_to_c, cc_gen]),
                                              nn.ModuleList([ccprev, w_cprev_to_c, cc_gen])]),
                      'rec_gen':   nn.ModuleList([nn.ModuleList([rcyorig_gen, yorig_to_zdiff, rczdiff_gen]),
                                              nn.ModuleList([rczdiff_gen, zdiff_zcom_wprev_to_w, rcw_gen]),
                                              nn.ModuleList([rczcom_gen, zdiff_zcom_wprev_to_w, rcw_gen]),
                                              nn.ModuleList([rcwprev, zdiff_zcom_wprev_to_w, rcw_gen]),
                                              nn.ModuleList([rcw_gen, w_cprev_to_c, rcc_gen]),
                                              nn.ModuleList([ccprev, w_cprev_to_c, rcc_gen])])}}, \
           {'clean': cc_gen, 'noise': nc_gen},\
           {'clean': cwid_inf, 'noise': nwid_inf},\
           {'clean': nn.ModuleList([nn.ModuleList([cw_sup, w_to_wid, cwid_gen])]),
            'noise': nn.ModuleList([nn.ModuleList([nw_sup, w_to_wid, nwid_gen])])}

