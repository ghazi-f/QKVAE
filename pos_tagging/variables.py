# This files serves defining latent variables for PoS tagging
import torch.nn as nn

from components.latent_variables import Categorical, Gaussian
from components import links
from components.criteria import IWLBo


class XInfer(Categorical):
    def __init__(self, h_params, word_embeddings, has_rep=True):
        if has_rep:
            repnet = nn.LSTM(word_embeddings.weight.shape[1], int(h_params.text_rep_h/2),
                             h_params.text_rep_l, batch_first=True, bidirectional=True, dropout=h_params.dropout)
        else:
            repnet = None
        super(XInfer, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                     h_params.vocab_ignore_index, markovian=not has_rep, repnet=repnet)


class XGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XGen, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                   h_params.vocab_ignore_index, markovian=True)


class XPrevGen(Categorical):
    def __init__(self, h_params, word_embeddings, has_rep=True):
        if has_rep:
            repnet = nn.LSTM(word_embeddings.weight.shape[1], h_params.text_rep_h, h_params.text_rep_l,
                             batch_first=True,
                             dropout=h_params.dropout)
        else:
            repnet = None
        super(XPrevGen, self).__init__(word_embeddings.weight.shape[0], 'x_prev', h_params.device, word_embeddings,
                                       h_params.vocab_ignore_index, markovian=not has_rep, is_placeholder=True,
                                       repnet=repnet, word_dropout=h_params.word_dropout)


class ZInfer(Gaussian):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses])
        super(ZInfer, self).__init__(h_params.z_size, 'z', h_params.device, markovian=h_params.markovian,
                                     stl=True, iw=iw, repnet=repnet)


class ZGen(Gaussian):
    def __init__(self, h_params, repnet, allow_prior=False):
        super(ZGen, self).__init__(h_params.z_size, 'z', h_params.device,
                                   markovian=h_params.markovian, allow_prior=allow_prior, repnet=repnet)


class ZSInfer(Gaussian):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses])
        super(ZSInfer, self).__init__(h_params.z_size, 'zs', h_params.device, markovian=h_params.markovian,
                                      stl=True, iw=iw, repnet=repnet)


class ZSGen(Gaussian):
    def __init__(self, h_params, repnet):
        super(ZSGen, self).__init__(h_params.z_size, 'zs', h_params.device,
                                    markovian=h_params.markovian, allow_prior=False, repnet=repnet)


class YEmbInfer(Gaussian):
    def __init__(self, h_params):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.piwo
        super(YEmbInfer, self).__init__(h_params.pos_embedding_dim, 'yemb', h_params.device,
                                        h_params.pos_ignore_index, markovian=True, stl=True, iw=iw)


class YEmbGen(Gaussian):
    def __init__(self, h_params):
        super(YEmbGen, self).__init__(h_params.pos_embedding_dim, 'yemb', h_params.device,
                                      h_params.pos_ignore_index, markovian=True)


class YvalInfer(Categorical):
    def __init__(self, h_params, pos_embeddings):
        super(YvalInfer, self).__init__(h_params.tag_size, 'y', h_params.device,  pos_embeddings,
                                        h_params.pos_ignore_index, markovian=True, is_placeholder=True)

