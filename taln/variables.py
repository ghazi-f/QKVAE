# This files serves defining latent variables for PoS tagging
import torch.nn as nn

from components.latent_variables import Categorical, Gaussian
from components import links
from components.criteria import IWLBo


class XInfer(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XInfer, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                     h_params.vocab_ignore_index, emb_batch_norm=h_params.emb_batch_norm)


class XGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XGen, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                   h_params.vocab_ignore_index,
                                   emb_batch_norm=h_params.emb_batch_norm)


class XPrevGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XPrevGen, self).__init__(word_embeddings.weight.shape[0], 'x_prev', h_params.device, word_embeddings,
                                       h_params.vocab_ignore_index, is_placeholder=True,
                                       word_dropout=h_params.word_dropout, emb_batch_norm=h_params.emb_batch_norm)


class ZInfer(Gaussian):
    def __init__(self, h_params, sequence_lv=True):
        super(ZInfer, self).__init__(h_params.z_size, 'z', h_params.device, stl=True, iw=False, sequence_lv=sequence_lv)


class ZGen(Gaussian):
    def __init__(self, h_params, allow_prior=False, sequence_lv=True):
        super(ZGen, self).__init__(h_params.z_size, 'z', h_params.device, allow_prior=allow_prior, sequence_lv=sequence_lv)
