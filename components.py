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
        self.items = self.parameters.items
        self.values = self.parameters.values
        self.keys = self.parameters.keys

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

    def prior_sample(self, sample_shape):
        # Returns a sample from P(z)
        sample = self.prior(**self.prior_params).sample(sample_shape)
        return sample, self.prior(**self.prior_params).log_prob(sample)

    def prior_log_prob(self, sample):
        prior_distrib = self.prior(**self.prior_params)
        return prior_distrib.log_prob(sample)


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

    @abc.abstractmethod
    def forward(self, z, x_prev=None):
        # The decoding function
        pass


class BaseCriterion(metaclass=abc.ABCMeta):
    def __init__(self, model, w):
        self.model = model
        self.h_params = model.h_params
        self.w = w

    @abc.abstractmethod
    def get_loss(self):
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
    parameters = {'loc': nn.Sequential(), 'scale_tril': torch.exp}

    def __init__(self, size, name, device):
        self.prior_loc = torch.zeros(size).to(device)
        self.prior_cov_tril = torch.eye(size).to(device)
        super(Gaussian, self).__init__(MultivariateNormal, size, {'loc': self.prior_loc,
                                                                  'scale_tril': self.prior_cov_tril},
                                       name)

    def infer(self, x_params):
        return x_params['loc']

    def posterior_sample(self, x_params):
        x_params['scale_tril'] = torch.diag_embed(x_params['scale_tril'])
        return super(Gaussian, self).posterior_sample(x_params)


class Categorical(BaseLatentVariable):
    parameters = {'logits': nn.Sequential()}

    def __init__(self, size, name, device):
        # IDEA: Try to implement "Direct Optimization through argmax"
        self.prior_logits = torch.ones(size).to(device)
        self.prior_temperature = torch.tensor([1.0]).to(device)
        super(Categorical, self).__init__(RelaxedOneHotCategorical, size, {'logits': self.prior_logits,
                                                                           'temperature': self.prior_temperature}, name)

    def infer(self, x_params):
        if 'temperature' not in x_params:
            x_params['temperature'] = torch.tensor([1.0])
        inferred = torch.argmax(x_params['logits'], dim=-1)
        return inferred, self.prior(x_params).log_prob(F.one_hot(inferred, x_params['logits'].shape[-1]))

    def posterior_sample(self, x_params):
        if 'temperature' not in x_params:
            x_params['temperature'] = self.prior_temperature
        return super(Categorical, self).posterior_sample(x_params)


# ======================================================================================================================
# ============================================== ENCODER CLASSES =======================================================
class GRUEncoder(BaseEncoder):
    def __init__(self, h_params, embeddings):
        super(GRUEncoder, self).__init__(h_params, embeddings)
        self.rnn = nn.GRU(h_params.embedding_dim, h_params.encoder_h, h_params.encoder_l, batch_first=True)
        self.project_z_prev = nn.Linear(sum([z.size for z in h_params.z_types]), h_params.encoder_h*h_params.encoder_l)

        # Creating a list of modules M that output latent variable parameters
        # Each Mi contains a list of modules Mij so that Mij outputs the parameter j of latent variable i
        self.hidden_to_z_params = nn.ModuleList([nn.ModuleDict({param: nn.Linear(h_params.encoder_h*h_params.encoder_l,
                                                                                 z_type.size)
                                                                for param in z_type.keys()})
                                                 for z_type in h_params.z_types])

    def forward(self, x, z_prev=None):
        embeds = self.word_embeddings(x)
        h_prev = self.project_z_prev(z_prev).view(self.h_params.encoder_l, x.shape[0], self.h_params.encoder_h) \
            if z_prev is not None else None
        rnn_out, h = self.rnn(embeds, hx=h_prev)
        # These parameters are those of q(zi|xi, z<i) for the current word i (even though it's called z_params and
        # not zi_params)
        reshaped_h = h.transpose(0, 1).reshape(x.shape[0], self.h_params.encoder_h*self.h_params.encoder_l)
        z_params = [{param: activation(h2zi_params[param](reshaped_h)) for param, activation in z_type.items()}
                    for h2zi_params, z_type in zip(self.hidden_to_z_params, self.h_params.z_types)]
        return z_params, rnn_out, h


# ======================================================================================================================
# ============================================== DECODER CLASSES========================================================

class GRUDecoder(BaseDecoder):
    def __init__(self, h_params, embeddings):
        super(GRUDecoder, self).__init__(h_params, embeddings)
        self.rnn = nn.GRU(sum([z.size for z in h_params.z_types]), h_params.decoder_h, h_params.decoder_l,
                          batch_first=True)
        self.rnn_to_embedding = torch.nn.Linear(h_params.decoder_h*h_params.decoder_l, h_params.embedding_dim)

        # this converts the concatenation of all z_i to the rnn input
        self.embedding_to_rnn = torch.nn.Linear(h_params.embedding_dim, h_params.decoder_h*h_params.decoder_l)


    def forward(self, z, x_prev=None):
        # The decoding
        h_prev =  self.embedding_to_rnn(x_prev).view(self.h_params.decoder_l, z.shape[0], self.h_params.decoder_h) \
            if x_prev is not None else None
        rnn_out, h = self.rnn(z, hx=h_prev)
        # These parameters are those of p(x_i|z_i, x<i) for the current word i (even though it's called x_params and
        # not xi_params)
        reshaped_h = h.transpose(0, 1).reshape(z.shape[0], self.h_params.decoder_h * self.h_params.decoder_l)
        embedded_rnn_out = self.rnn_to_embedding(reshaped_h)
        x_params = torch.matmul(embedded_rnn_out, self.word_embeddings.weight.transpose(dim0=0, dim1=1))

        return x_params, rnn_out, h


# ======================================================================================================================
# ============================================== CRITERION CLASSES =====================================================

class Supervision(BaseCriterion):
    def __init__(self, model, w, supervised_z_index):
        super(Supervision, self).__init__(model, w)
        self.supervised_z_index = supervised_z_index

        criterion_params = {'ignore_index': self.h_params.target_ignore_index}
        if self.h_params.weight_prediction:
            counts = [model.tag_index.freqs[w] for w in self.model.tag_index.itos]
            freqs = torch.Tensor([1/c if c != 0 else 0 for c in counts]).to(self.h_params.device)
            criterion_params['weight'] = freqs/torch.sum(freqs)
        self.criterion = nn.CrossEntropyLoss(**criterion_params)

    def get_loss(self):
        num_classes = self.h_params.z_types[self.supervised_z_index].size
        predictions = torch.stack([z_i_params[self.supervised_z_index]['logits']
                                   for z_i_params in self.model.Z_params]).transpose(0, 1).reshape(-1, num_classes)
        target = self.model.Y.view(-1)

        return self.criterion(predictions, target)

    def _common_metrics(self):
        ce = self.get_loss()
        with torch.no_grad():
            num_classes = self.h_params.z_types[self.supervised_z_index].size
            predictions = torch.stack([z_i_params[self.supervised_z_index]['logits']
                                       for z_i_params in self.model.Z_params]).transpose(0, 1).reshape(-1, num_classes)
            prediction_mask = (self.model.Y.view(-1) != self.h_params.target_ignore_index).float()
            accuracy = torch.sum((torch.argmax(predictions, dim=-1) == self.model.Y.view(-1)).float()*prediction_mask)
            accuracy /= torch.sum(prediction_mask)
        return {'/supervision_CE': ce, '/supervision_accuracy': accuracy}

    def _train_metrics(self):
        pass
        return {}

    def _test_metrics(self):
        return {}


class Reconstruction(BaseCriterion):
    def __init__(self, model, w):
        super(Reconstruction, self).__init__(model, w)
        criterion_params = {'ignore_index': self.h_params.token_ignore_index}
        if self.h_params.weight_reconstruction:
            freqs = 1/torch.Tensor([model.vocab_index.freqs[w] for w in self.model.vocab_index.itos]).to(self.h_params.device)
            criterion_params['weight'] = freqs/torch.sum(freqs)
        self.criterion = nn.CrossEntropyLoss(**criterion_params)

    def get_loss(self):
        self.criterion(torch.cat(self.model.X_hat_params), F.one_hot(self.model.X))
        return self.criterion(torch.cat(self.model.X_hat_params), F.one_hot(self.model.X))

    def _common_metrics(self):
        return {'/reconstruction_CE': self.get_loss()}

    def _train_metrics(self):
        pass
        return {}

    def _test_metrics(self):
        return {}


class ELBo(BaseCriterion):
    # This is actually Sticking The Landing (STL) ELBo, and not the standard one (it estimates the same quantity anyway)
    def __init__(self, model, w):
        super(ELBo, self).__init__(model, w)
        criterion_params = {'ignore_index': self.h_params.token_ignore_index, 'reduction': 'none'}
        if self.h_params.weight_reconstruction:
            counts = [model.vocab_index.freqs[w] for w in self.model.vocab_index.itos]
            freqs = torch.Tensor([1/c if c != 0 else 0 for c in counts]).to(self.h_params.device)
            criterion_params['weight'] = freqs/torch.sum(freqs)
        self.criterion = nn.CrossEntropyLoss(**criterion_params)
        if self.h_params.weight_reconstruction:
            criterion_params.pop('weight')
        self._unweighted_criterion = nn.CrossEntropyLoss(**criterion_params)
        self.log_p_xIz = None
        self.log_p_z = None
        self.log_q_zIx = None

    def get_loss(self, unweighted=False):
        # This probability is shaped like [batch_size]
        vocab_size = self.h_params.vocab_size
        criterion = self._unweighted_criterion if unweighted else self.criterion
        self.log_p_xIz = - criterion(self.model.X_hat_params.reshape(-1, vocab_size),
                                          self.model.X.view(-1)).view(self.model.X.shape)

        # The following 2 probabilities are shaped [z_types, batch_size]
        self.log_p_z = []
        for z_i_samples in self.model.Z_samples:
            current_z_beginning = 0
            running_log_prob = []
            for z_type in self.model.h_params.z_types:
                running_log_prob.append(z_type.prior_log_prob(z_i_samples[:,
                                                              current_z_beginning: current_z_beginning+z_type.size]))
                current_z_beginning += z_type.size
            self.log_p_z.append(torch.stack(running_log_prob))
        self.sequence_mask = (self.model.X != self.h_params.token_ignore_index).float()
        self.valid_n_samples = torch.sum(self.sequence_mask)
        self.log_p_z_i = torch.stack(self.log_p_z).transpose(0, 1) * self.sequence_mask.unsqueeze(0)
        self.log_p_z = torch.sum(self.log_p_z_i, dim=0)
        self.log_q_z_iIx = self.model.Z_log_probas.transpose(0, 1) * self.sequence_mask.unsqueeze(0)
        self.log_q_zIx = torch.sum(self.log_q_z_iIx, dim=0)
        return - torch.sum(self.log_p_xIz + self.log_p_z - self.log_q_zIx, dim=(0, 1))/self.valid_n_samples

    def _common_metrics(self):
        current_elbo = - self.get_loss(unweighted=True)
        return {'/ELBo': current_elbo/self.valid_n_samples,
                '/reconstruction_LL': torch.sum(self.log_p_xIz)/self.valid_n_samples,
                **{'/KL(q({}|x)Ip({}))'.format(z.name, z.name):
                   torch.sum(log_qz_iIx - log_pz_i)/self.valid_n_samples
                   for z, log_qz_iIx, log_pz_i in zip(self.model.h_params.z_types,
                                                      self.log_q_z_iIx,
                                                      self.log_p_z_i)}}

    def _train_metrics(self):
        return {}

    def _test_metrics(self):
        return {}


class IWLBo(ELBo):
    # This is actually DReG IWLBo and not the standard one (it estimates the same quantity anyway)
    def __init__(self, model, w):
        assert isinstance(model, ImportanceWeightedAEMixin), "To use IWLBo, your model has to inherit from the " \
                                                             "ImportanceWeightedAEMixin class"
        super(ELBo, self).__init__(model, w)
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.h_params.token_ignore_index)
        self.log_p_xIz = None
        self.log_p_z = None
        self.log_q_zIx = None

    def get_loss(self):

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
