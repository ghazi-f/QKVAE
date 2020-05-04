# This files serves defining latent variables for PoS tagging
import torch.nn as nn

from components.latent_variables import Categorical, Gaussian
from components import links

MARKOVIAN = False


class XInfer(Categorical):
    def __init__(self, h_params, word_embeddings):
        repnet = nn.LSTM(word_embeddings.weight.shape[1], int(word_embeddings.weight.shape[1]/2), 5, batch_first=True,
                        bidirectional=True)
        super(XInfer, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                     h_params.vocab_ignore_index, markovian=MARKOVIAN, repnet=repnet)


class XGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XGen, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                   h_params.vocab_ignore_index, markovian=True)


class XPrevGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        repnet = nn.LSTM(word_embeddings.weight.shape[1], int(word_embeddings.weight.shape[1]/2), 5, batch_first=True)
        super(XPrevGen, self).__init__(word_embeddings.weight.shape[0], 'x_prev', h_params.device, word_embeddings,
                                       h_params.vocab_ignore_index, markovian=False, is_placeholder=True, repnet=repnet)


class ZInfer(Gaussian):
    def __init__(self, h_params):
        super(ZInfer, self).__init__(h_params.z_size, 'z', h_params.device, markovian=MARKOVIAN, inv_seq=True, stl=True)


class ZGen(Gaussian):
    def __init__(self, h_params):
        #prior_seq_link = links.GRULink(h_params.z_size, h_params.z_size*5, h_params.z_size, 4, Gaussian.parameters)
        super(ZGen, self).__init__(h_params.z_size, 'z', h_params.device, #prior_sequential_link=prior_seq_link,
                                   markovian=MARKOVIAN, allow_prior=False)


class YInfer(Categorical):
    def __init__(self, h_params, pos_embeddings):
        super(YInfer, self).__init__(pos_embeddings.weight.shape[0], 'y', h_params.device, pos_embeddings,
                                     h_params.pos_ignore_index, markovian=MARKOVIAN, stl=True)


class YGen(Categorical):
    def __init__(self, h_params, pos_embeddings):
        super(YGen, self).__init__(pos_embeddings.weight.shape[0], 'y', h_params.device, pos_embeddings,
                                   h_params.pos_ignore_index, markovian=MARKOVIAN)

