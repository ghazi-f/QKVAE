import math
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import MultivariateNormal, RelaxedOneHotCategorical

from models import *


# ============================================== BASE CLASSES ==========================================================
class BaseLatentVariable(nn.Module, metaclass=abc.ABCMeta):
    # Define the parameters with their corresponding layer activation function
    # NOTE: identity function behaviour can be produced with torch.nn.Sequential() as an activation function
    parameters = {}

    def __init__(self, prior, size, prior_params, name):
        super(BaseLatentVariable, self).__init__()
        assert len(self.parameters) > 0
        self.prior = prior
        self.size = size
        self.prior_params = prior_params
        self.name = name

    @abc.abstractmethod
    def infer(self, x_params):
        # Returns z = argmax P(z|x)
        pass

    def posterior_sample(self, x_params):
        # Returns a sample from P(z|x). x is a dictionary containing the distribution's parameters.
        if self.prior.has_rsample:
            sample = self.prior(**x_params).rsample()
            # Applying STL
            detached_prior = self.prior(**{k: v.detach() for k, v in x_params.items()})
            return sample, detached_prior.log_prob(sample)
        else:
            raise NotImplementedError('Distribution {} has no reparametrized sampling method.')

    def prior_sample(self):
        # Returns a sample from P(z)
        sample = self.prior(**self.prior_params).sample()
        return sample, self.prior(**self.prior_params).log_prob(sample)


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

    def forward(self, z, is_training, x_prev=None):
        # The decoding function
        if is_training:
            return self._train_decoding(z, x_prev=None)
        else:
            return self._test_decoding(z, x_prev=None)

    @abc.abstractmethod
    def _train_decoding(self, z, x_prev=None):
        # The decoding at train time
        pass

    @abc.abstractmethod
    def _test_decoding(self, z, x_prev=None):
        # The decoding at test time
        pass


class BaseCriterion(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, model: BaseSSAE, w):
        super(BaseCriterion, self).__init__()
        self.model = model
        self.h_params = model.h_params
        self.w = w

    @abc.abstractmethod
    def forward(self):
        # The loss function
        pass

    def metrics(self, is_training):
        if is_training:
            return {**(self._train_metrics()), **(self._common_metrics())}.items()
        else:
            return {**(self._test_metrics()), **(self._common_metrics())}.items()

    @abc.abstractmethod
    def _common_metrics(self):
        pass

    def _train_metrics(self):
        return {}

    def _test_metrics(self):
        return {}


# ======================================================================================================================
# ================================================ LATENT VARIABLE CLASSES =============================================

class Gaussian(BaseLatentVariable):
    parameters = {'loc': nn.Sequential, 'covariance_matrix': torch.exp}

    def __init__(self, size, name):
        super(Gaussian, self).__init__(MultivariateNormal, size, {'loc': torch.zeros(size),
                                                                  'covariance_matrix': torch.eye(size)}, name)

    def infer(self, x_params):
        return x_params['loc']


class Categorical(BaseLatentVariable):
    parameters = {'logits': nn.Sequential}

    def __init__(self, size, name):
        # IDEA: Try to implement "Direct Optimization through argmax"
        super(Categorical, self).__init__(RelaxedOneHotCategorical, size, {'logits': torch.ones(size),
                                                                           'temperature': torch.tensor([1.0])}, name)

    def infer(self, x_params):
        if 'temperature' not in x_params:
            x_params['temperature'] = torch.tensor([1.0])
        inferred = torch.argmax(x_params['logits'], dim=-1)
        return inferred, self.prior(x_params).log_prob(F.one_hot(inferred, x_params['logits'].shape[-1]))

    def posterior_sample(self, x_params):
        if 'temperature' not in x_params:
            x_params['temperature'] = torch.tensor([1.0])
        return super(Categorical, self).posterior_sample(x_params)


# ======================================================================================================================
# ============================================== ENCODER CLASSES =======================================================
class GRUEncoder(BaseEncoder):
    def __init__(self, h_params, embeddings):
        super(GRUEncoder, self).__init__(h_params, embeddings)
        self.rnn = nn.GRU(h_params.embedding_dim, h_params.encoder_h, h_params.encoder_l, batch_first=True)

        # Creating a list of modules M that output latent variable parameters
        # Each Mi contains a list of modules Mij so that Mij outputs the parameter j of latent variable i
        self.hidden_to_z_params = nn.ModuleList([nn.ModuleDict({param: activation(nn.Linear(h_params.encoder_h,
                                                                                            z_type.size))
                                                                for param, activation in z_type.items()})
                                                 for z_type in h_params.z_types])

    def forward(self, x, z_prev=None):

        embeds = self.word_embeddings(x)
        rnn_out, h = self.rnn(embeds, hx=z_prev)
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

    def _train_decoding(self, z, x_prev=None):
        # The decoding at train time (teacher-forced)
        rnn_out, h = self.rnn(z, hx=x_prev)
        # These parameters are sampled from p(x_i|z_i, x<i) for the current word i (even though it's called x_params and
        # not xi_params)
        x_params = torch.matmul(rnn_out, self.word_embeddings.weight.transpose())

        return x_params, rnn_out, h

    def _test_decoding(self, z, x_prev=None):
        # The decoding at test time (non-teacher-forced)
        with torch.no_grad():
            pass


# ======================================================================================================================
# ============================================== CRITERION CLASSES =====================================================

class Supervision(BaseCriterion):
    def __init__(self, model, w, supervised_z_index):
        super(Supervision, self).__init__(model, w)
        self.supervised_z_index = supervised_z_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self):
        return self.criterion(torch.cat([z[self.supervised_z_index]['logits'] for z in self.model.Z]),
                              F.one_hot(self.model.Y))

    def _common_metrics(self):
        return {'supervision_CE': self.forward()}

    def _train_metrics(self):
        pass
        return {}

    def _test_metrics(self):
        return {}


class Reconstruction(BaseCriterion):
    def __init__(self, model, w):
        super(Reconstruction, self).__init__(model, w)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self):
        return self.criterion(torch.cat(self.model.X_hat_params), F.one_hot(self.model.X))

    def _common_metrics(self):
        return {'reconstruction_CE': self.forward()}

    def _train_metrics(self):
        pass
        return {}

    def _test_metrics(self):
        return {}


class ELBo(BaseCriterion):
    # This is actually Sticking The Landing (STL) ELBo, and not the standard one
    def __init__(self, model, w):
        super(ELBo, self).__init__(model, w)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.log_p_xIz = None
        self.log_p_z = None
        self.log_q_zIx = None

    def forward(self):
        # This probability is shaped like [batch_size]
        self.log_p_xIz = self.criterion(torch.cat(self.model.X_hat_params), F.one_hot(self.model.X))

        # The following 2 probabilities are shaped [z_types, batch_size]
        self.log_p_z = [z_type.prior.log_prob(z_i_samples)
                        for z_type, z_i_samples in zip(self.model.h_params.z_types,
                                                       self.model.Z_samples)]
        self.log_q_zIx = self.model.Z_log_probas

        return torch.mean(self.log_p_xIz + torch.sum(torch.cat(self.log_p_z) - torch.cat(self.log_q_zIx), dim=0), dim=0)

    def _common_metrics(self):
        current_elbo = self.forward()
        return {'ELBo': current_elbo, 'log_px|z,y': torch.mean(self.log_p_xIz),
                **{'log_q{}|x'.format(z.name): torch.mean(log_qziIx) for z, log_qziIx in zip(self.model.h_params.z_types,
                                                                                             self.log_q_zIx)},
                **{'log_p{}'.format(z.name): torch.mean(log_pz) for z, log_pz in zip(self.model.h_params.z_types,
                                                                                             self.log_p_z)}}

    def _train_metrics(self):
        return {}

    def _test_metrics(self):
        return {}


class IWLBo(ELBo):
    # This is actually DReG IWLBo and not the standard one
    def __init__(self, model, w):
        assert isinstance(model, ImportanceWeightedAEMixin), "To use IWLBo, your model has to inherit from the " \
                                                             "ImportanceWeightedAEMixin class"
        super(ELBo, self).__init__(model, w)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.log_p_xIz = None
        self.log_p_z = None
        self.log_q_zIx = None

    def forward(self):

        # This probability is shaped like [iw_samples, batch_size]
        self.log_p_xIz = torch.cat([self.criterion(xhat_p_i, F.one_hot(self.model.X))
                                    for xhat_p_i in torch.cat(self.model.X_hat_params).view(self.model.iw_samples, -1)])

        # The following 2 probabilities are shaped [z_types, iw_samples, batch_size]
        self.log_p_z = [z_type.prior.log_prob(z_i_samples).view(self.model.iw_samples, -1)
                                      for z_type, z_i_samples in zip(self.model.h_params.z_types,
                                                                     self.model.Z_samples)]
        self.log_q_zIx = self.model.Z_log_probas

        return torch.mean(self.log_p_xIz + torch.sum(self.log_p_z - self.log_q_zIx, dim=0), dim=0)

    def _common_metrics(self):
        pass

    def _train_metrics(self):
        return {}

    def _test_metrics(self):
        return {}


# ======================================================================================================================

