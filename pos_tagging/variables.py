# This files serves defining latent variables for PoS tagging
import torch.nn as nn

from components.latent_variables import Categorical, Gaussian
from components import links
from components.criteria import IWLBo

MARKOVIAN = False


class XInfer(Categorical):
    def __init__(self, h_params, word_embeddings):
        repnet = nn.LSTM(word_embeddings.weight.shape[1], int(h_params.text_rep_h/2),
                         h_params.text_rep_l, batch_first=True, bidirectional=True, dropout=h_params.dropout)
        super(XInfer, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                     h_params.vocab_ignore_index, markovian=MARKOVIAN, repnet=repnet)


class XGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XGen, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                   h_params.vocab_ignore_index, markovian=True)


class XPrevGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        repnet = nn.LSTM(word_embeddings.weight.shape[1], int(h_params.text_rep_h), h_params.text_rep_l,
                         batch_first=True,
                         dropout=h_params.dropout)
        super(XPrevGen, self).__init__(word_embeddings.weight.shape[0], 'x_prev', h_params.device, word_embeddings,
                                       h_params.vocab_ignore_index, markovian=False, is_placeholder=True, repnet=repnet)


class ZInfer(Gaussian):
    def __init__(self, h_params):
        iw = any([l == IWLBo for l in h_params.losses])
        super(ZInfer, self).__init__(h_params.z_size, 'z', h_params.device, markovian=MARKOVIAN, inv_seq=True, stl=True,
                                     iw=iw)


class ZGen(Gaussian):
    def __init__(self, h_params):
        #prior_seq_link = links.GRULink(h_params.z_size, h_params.z_size*5, h_params.z_size, 4, Gaussian.parameters)
        super(ZGen, self).__init__(h_params.z_size, 'z', h_params.device, #prior_sequential_link=prior_seq_link,
                                   markovian=MARKOVIAN, allow_prior=False)


class YEmbInfer(Gaussian):
    def __init__(self, h_params):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.piwo
        super(YEmbInfer, self).__init__(h_params.pos_embedding_dim, 'yemb', h_params.device,
                                        h_params.pos_ignore_index, markovian=MARKOVIAN, stl=True, iw=iw)


class YEmbGen(Gaussian):
    def __init__(self, h_params):
        super(YEmbGen, self).__init__(h_params.pos_embedding_dim, 'yemb', h_params.device,
                                      h_params.pos_ignore_index, markovian=MARKOVIAN)


class YvalInfer(Categorical):
    def __init__(self, h_params, pos_embeddings):
        super(YvalInfer, self).__init__(h_params.tag_size, 'y', h_params.device,  pos_embeddings,
                                        h_params.pos_ignore_index, markovian=MARKOVIAN, is_placeholder=True)

