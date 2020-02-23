import math
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models import *


# ============================================== BASE CLASSES ==========================================================
class BaseEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, h_params, embeddings):
        super(BaseEncoder, self).__init__()
        self.h_params = h_params
        self.word_embeddings = embeddings

    @abc.abstractmethod
    def forward(self, x):
        # The encoding function
        pass


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, h_params, embeddings):
        super(BaseDecoder, self).__init__()
        self.h_params = h_params
        self.word_embeddings = embeddings

    def forward(self, z, is_training):
        # The decoding function
        if is_training:
            return self._train_decoding(z)
        else:
            return self._test_decoding(z)

    @abc.abstractmethod
    def _train_decoding(self, z):
        # The decoding at train time
        pass

    @abc.abstractmethod
    def _test_decoding(self, z):
        # The decoding at test time
        pass


class BaseCriterion(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, model: BaseSSAE, w):
        super(BaseCriterion, self).__init__()
        self.model = model
        self.h_params = model.h_params
        self.w = w

    @abc.abstractmethod
    def forward(self, samples):
        # The loss function
        pass

    def metrics(self, samples=None):
        if samples is not None:
            return {**(self._train_metrics()), **(self._common_metrics())}
        else:
            return {**(self._test_metrics(samples)), **(self._common_metrics(samples))}

    @abc.abstractmethod
    def _common_metrics(self, samples=None):
        pass

    def _train_metrics(self):
        return {}

    def _test_metrics(self, samples):
        return {}


# ======================================================================================================================
# ============================================== ENCODER CLASSES =======================================================

class GRUEncoder(BaseEncoder):
    def __init__(self, h_params, embeddings):
        super(GRUEncoder, self).__init__(h_params, embeddings)
        self.rnn = nn.GRU(h_params.embedding_dim, h_params.encoder_h, h_params.encoder_l, batch_first=True)

        # Creating a list of modules M that output latent variable parameters
        # Each Mi contains a list of modules Mij so that Mij outputs the parameter j of latent variable i
        self.hidden_to_z_params = nn.ModuleList([nn.ModuleDict({param: activation(nn.Linear(h_params.encoder_h, size))
                                                                for param, activation in z_type.items()})
                                                 for z_type, size in zip(h_params.z_types, h_params.z_sizes)])

    def forward(self, x, z_prev=None):

        embeds = self.word_embeddings(x)
        rnn_out, h = self.rnn(embeds, )
        # These parameters are sampled from q(zi|xi, z<i) for the current word i (even though it's called z_params and
        # not zi_params)
        z_params = [{param: h2zi_param(rnn_out) for param, h2zi_param in h2zi_params.items()}
                    for h2zi_params in self.hidden_to_zs]
        return z_params, rnn_out, h


# ======================================================================================================================
# ============================================== DECODER CLASSES========================================================

class GRUDecoder(BaseDecoder):
    # Go read https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
    # TODO: tie weights between encoding and decoding
    def __init__(self, h_params, embeddings):
        super(GRUDecoder, self).__init__(h_params, embeddings)
        self.rnn = nn.GRU(sum(h_params.z_sizes), h_params.decoder_h, h_params.encoder_l, batch_first=True)

        # this converts the concatenation of all z_i to the rnn input

    def _train_decoding(self, z):
        # The decoding at train time (teacher-forced)
        rnn_out, h = self.rnn(z)
        # These parameters are sampled from p(x_i|z_i, x<i) for the current word i (even though it's called x_params and
        # not xi_params)
        x_params = torch.matmul(rnn_out, self.word_embeddings.weight.transpose())

        return x_params, rnn_out, h

    def _test_decoding(self, z):
        # The decoding at test time (non-teacher-forced)
        with torch.no_grad():
            pass


# ======================================================================================================================
# ============================================== CRITERION CLASSES =====================================================

class Supervision(BaseCriterion):
    def __init__(self, model, w):
        super(Supervision, self).__init__(model, w)

    def forward(self, samples):
        pass

    def _common_metrics(self, samples=None):
        pass

    def _train_metrics(self):
        pass

    def _test_metrics(self, samples):
        with torch.no_grad():
            self.forward(samples)
            pass


class Reconstruction(BaseCriterion):
    def __init__(self, model, w):
        super(Reconstruction, self).__init__(model, w)

    def forward(self, samples):
        pass

    def _common_metrics(self, samples=None):
        pass

    def _train_metrics(self):
        pass

    def _test_metrics(self, samples):
        with torch.no_grad():
            self.forward(samples)
            pass


class ELBo(Reconstruction):
    def __init__(self, model, w):
        super(ELBo, self).__init__(model, w)

    def forward(self, samples):
        pass

    def _common_metrics(self, samples=None):
        pass

    def _train_metrics(self):
        pass

    def _test_metrics(self, samples):
        with torch.no_grad():
            self.forward(samples)
            pass


class IWLBo(ELBo):
    def __init__(self, model, w):
        super(ELBo, self).__init__(model, w)

    def forward(self, samples):
        pass

    def _common_metrics(self, samples=None):
        pass

    def _train_metrics(self):
        pass

    def _test_metrics(self, samples):
        with torch.no_grad():
            self.forward(samples)
            pass


# ======================================================================================================================

