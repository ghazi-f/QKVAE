# This files serves defining latent variables for normalization
import torch.nn as nn
from components.latent_variables import Categorical, Gaussian


class CInfer(Categorical):
    def __init__(self, h_params, embeddings):
        super(CInfer, self).__init__(embeddings.weight.shape[0], 'c', h_params.device, embeddings,
                                     h_params.c_ignore_index, sub_lvl_size=h_params.c_max_len)


class CGen(Categorical):
    def __init__(self, h_params, embeddings):
        super(CGen, self).__init__(embeddings.weight.shape[0], 'c', h_params.device, embeddings,
                                   h_params.c_ignore_index, sub_lvl_size=h_params.c_max_len)


class CPrevGen(Categorical):
    def __init__(self, h_params, embeddings):
        super(CPrevGen, self).__init__(embeddings.weight.shape[0], 'c_prev', h_params.device, embeddings,
                                       h_params.c_ignore_index, is_placeholder=True,
                                       word_dropout=h_params.char_dropout, sub_lvl_size=h_params.c_max_len)


class WidInfer(Categorical):
    def __init__(self, h_params, embeddings):
        super(WidInfer, self).__init__(embeddings.weight.shape[0], 'wid', h_params.device, embeddings,
                                       h_params.w_ignore_index, is_placeholder=True)


class WidGen(Categorical):
    def __init__(self, h_params, embeddings):
        super(WidGen, self).__init__(embeddings.weight.shape[0], 'wid', h_params.device, embeddings,
                                     h_params.w_ignore_index, is_placeholder=True)


class YorigInfer(Categorical):
    def __init__(self, h_params, embeddings):
        super(YorigInfer, self).__init__(embeddings.weight.shape[0], 'yorig', h_params.device, embeddings,
                                         h_params.y_ignore_index)


class YorigGen(Categorical):
    def __init__(self, h_params, embeddings):
        super(YorigGen, self).__init__(embeddings.weight.shape[0], 'yorig', h_params.device, embeddings,
                                       h_params.y_ignore_index, allow_prior=True)


class WInfer(Gaussian):
    def __init__(self, h_params):
        super(WInfer, self).__init__(h_params.w_embedding_dim, 'w', h_params.device,
                                        stl=True)


class WGen(Gaussian):
    def __init__(self, h_params):
        super(WGen, self).__init__(h_params.w_embedding_dim, 'w', h_params.device,
                                      stl=True)


class WPrevGen(Gaussian):
    def __init__(self, h_params):
        super(WPrevGen, self).__init__(h_params.w_embedding_dim, 'w_prev', h_params.device, is_placeholder=True)


class ZcomInfer(Gaussian):
    def __init__(self, h_params):
        super(ZcomInfer, self).__init__(h_params.zcom_size, 'zcom', h_params.device,
                                        stl=True, sequence_lv=True)


class ZcomGen(Gaussian):
    def __init__(self, h_params):
        super(ZcomGen, self).__init__(h_params.zcom_size, 'zcom', h_params.device,
                                      stl=True, sequence_lv=True, allow_prior=False)


class ZdiffInfer(Gaussian):
    def __init__(self, h_params):
        super(ZdiffInfer, self).__init__(h_params.zdiff_size, 'zdiff', h_params.device,
                                         stl=True, sequence_lv=True)


class ZdiffGen(Gaussian):
    def __init__(self, h_params):
        super(ZdiffGen, self).__init__(h_params.zdiff_size, 'zdiff', h_params.device,
                                       stl=True, sequence_lv=True)

