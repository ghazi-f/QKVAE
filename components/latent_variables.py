import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, RelaxedOneHotCategorical

from components.links import SequentialLink


# ============================================== BASE CLASS ============================================================

class BaseLatentVariable(nn.Module, metaclass=abc.ABCMeta):
    # Define the parameters with their corresponding layer activation function
    # NOTE: identity function behaviour can be produced with torch.nn.Sequential() as an activation function
    parameters = {}

    def __init__(self, prior, size, prior_params, name, prior_sequential_link=None, posterior=None, markovian=True,
                 allow_prior=False, is_placeholder=False, inv_seq=False, stl=False, repnet=None):
        # IDEA: Lock latent variable behaviour according to it's role in the bayesian network
        super(BaseLatentVariable, self).__init__()
        assert len(self.parameters) > 0
        self.prior_sequential_link = prior_sequential_link
        self.prior = prior
        self.allow_prior = allow_prior
        self.posterior = posterior or prior
        self.size = size
        self.is_placeholder = is_placeholder
        self.inv_seq = inv_seq
        self.stl = stl

        # If the representation is non Markovian an LSTM is added to represent the variable
        if markovian:
            self.rep_net = None
        else:
            if isinstance(self, Categorical):
                # Wait for the Categorical constructor to instanciate a GRU with the right size
                self.rep_net = True
            else:
                self.rep_net = repnet or nn.GRU(self.size, self.size, 1, batch_first=True)

        self.prior_params = prior_params
        self.prior_samples = None
        self.prior_reps = None
        self.prior_log_probas = None

        self.name = name
        self.items = self.parameters.items
        self.values = self.parameters.values
        self.keys = self.parameters.keys

        self.post_params = None
        self.post_samples = None
        self.post_reps = None
        self.post_log_probas = None
        self.post_gt_log_probas = None

    @abc.abstractmethod
    def infer(self, x_params):
        # Returns z = argmax P(z|x)
        # TODO: handle SMC case
        pass

    def posterior_sample(self, x_params):
        # Returns a sample from P(z|x). x is a dictionary containing the distribution's parameters.
        if self.posterior.has_rsample:
            sample = self.posterior(**x_params).rsample()
            # Applying STL
            if self.stl:
                prior = self.posterior(**{k: v.detach() for k, v in x_params.items()})
            else:
                prior = self.posterior(**{k: v for k, v in x_params.items()})
            return sample, prior.log_prob(sample)
        else:
            raise NotImplementedError('Distribution {} has no reparametrized sampling method.')

    def prior_sample(self, sample_shape):
        assert not self.inv_seq, "Reversed priors are still not permitted"
        assert self.allow_prior, "{} Doesn't allow for a prior".format(self)
        if self.prior.has_rsample:
            # Returns a sample from P(z)
            if self.prior_sequential_link is not None:
                self.prior_samples = []
                self.prior_log_probas = []
                self.prior_reps = []
                prev_rep = None
                prior_params_i = self.prior_params
                for i in range(sample_shape[-1]):
                    prior_distrib = self.prior(**prior_params_i)
                    self.prior_samples.append(prior_distrib.sample(sample_shape[:-1] if i == 0 else ()).squeeze())
                    self.prior_log_probas.append(prior_distrib.log_prob(self.prior_samples[-1]))
                    self.prior_reps.append(self.rep(self.prior_samples[-1], prev_rep=prev_rep))
                    prev_rep = self.prior_reps[-1]
                    prior_params_i = self.prior_sequential_link(prev_rep, prev_rep)
                    if isinstance(self, Gaussian):
                        prior_params_i['scale_tril'] = torch.diag_embed(prior_params_i['scale_tril'])
                    elif isinstance(self, Categorical):
                        prior_params_i = {**prior_params_i, **{'temperature': self.prior_temperature}}
                self.prior_samples = torch.stack(self.prior_samples, dim=-2)
                self.prior_log_probas = torch.stack(self.prior_log_probas, dim=-1)

            else:
                prior_distrib = self.prior(**self.prior_params)
                self.prior_samples = prior_distrib.sample(sample_shape)
                self.prior_log_probas = prior_distrib.log_prob(self.prior_samples)
                self.prior_reps = self.rep(self.prior_samples, step_wise=False)

            return self.prior_samples, self.prior_log_probas

        else:
            raise NotImplementedError('Distribution {} has no reparametrized sampling method.')

    def prior_log_prob(self, sample):
        assert not self.inv_seq, "Reversed priors are still not permitted"
        assert self.allow_prior, "{} Doesn't allow for a prior".format(self)
        assert isinstance(self, Gaussian), "This is still not implemented for latent variables that are {} " \
                                           "and not Gaussian".format(repr(self))
        if sample.dtype == torch.long and isinstance(self, Categorical):
            sample = F.one_hot(sample, self.size).float()
        if self.prior_sequential_link is not None:
            sample_rep = self.rep(sample, step_wise=False)
            prior_params_i = {k:v.repeat(list(sample.shape[:-2])+[1]*v.ndim) for k, v in self.prior_params.items()}
            prior_log_probas = []
            z_params = {param: [] for param in self.parameters}
            for sample_i, sample_rep_i in zip(sample.transpose(0, -2), sample_rep.transpose(0, -2)):
                for k, v in prior_params_i.items():
                    z_params[k].append(prior_params_i[k])
                prior_distrib = self.prior(**prior_params_i)
                prior_log_probas.append(prior_distrib.log_prob(sample_i))
                prior_params_i = self.prior_sequential_link(sample_rep_i, sample_rep_i)
                prior_params_i['scale_tril'] = torch.diag_embed(prior_params_i['scale_tril'])
            for k, v in z_params.items():
                z_params[k] = torch.stack(v, dim=-1-self.prior_params[k].ndim)
            self.post_params = z_params
            return torch.stack(prior_log_probas, dim=-1)

        else:
            prior_distrib = self.prior(**self.prior_params)
            return prior_distrib.log_prob(sample)

    def forward(self, link_approximator, inputs, prior=None, gt_samples=None):
        if isinstance(link_approximator, SequentialLink):
            self._sequential_forward(link_approximator, inputs, prior, gt_samples)
        else:
            self._forward(link_approximator, inputs, prior, gt_samples)

    @abc.abstractmethod
    def rep(self, samples, step_wise=True, prev_rep=None):
        # Get sample representations
        pass

    def _sequential_forward(self, link_approximator, inputs, prior=None, gt_samples=None):
        if self.inv_seq:
            inputs = torch.flip(inputs, dims=[-2])
            if gt_samples is not None:
                gt_samples = torch.flip(gt_samples, dims=[-2])
        z_params = {param: [] for param in self.prior_params}
        z_log_probas = []
        z_samples = []
        prev_zs = [prior] if prior else []
        if gt_samples is not None:
            prev_zs = self.rep(gt_samples, step_wise=False).transpose(0, -2)
            if gt_samples.dtype == torch.long and isinstance(self, Categorical):
                gt_samples = F.one_hot(gt_samples, self.size).float()
                gt_samples = gt_samples.transpose(0, -2)
            else:
                gt_samples = gt_samples.transpose(0, -2)
            gt_log_probas = []
        z_reps = []

        for x in inputs.transpose(0, -2):
            # Inputs are unsqueezed to have single element time-step dimension, while previous outputs are unsqueezed to
            # have a single element num_layers*num_directions dimension (to be changed for multilayer GRUs)
            prev_z = prev_zs[len(z_reps)-1] if len(z_reps) else None
            z_params_i = link_approximator(x, prev_z)
            posterior, posterior_log_prob = self.posterior_sample(z_params_i)
            z_samples.append(posterior)
            z_log_probas.append(posterior_log_prob)
            z_reps.append(self.rep(posterior, prev_rep=prev_z))
            if isinstance(self, Gaussian):
                z_params_i['scale_tril'] = torch.diag_embed(z_params_i['scale_tril'])
            elif isinstance(self, Categorical):
                z_params_i = {**z_params_i, **{'temperature': self.prior_temperature}}
            else:
                raise NotImplementedError("This function still hasn't been implemented for variables other than "
                                          "Gaussians and Categoricals")
            for k, v in z_params_i.items():
                z_params[k].append(z_params_i[k])
            if gt_samples is not None:
                if gt_samples.dtype == torch.long:
                    print('Is long')
                    prob = torch.sum(z_params_i['logits'] * gt_samples[len(z_reps)-1], dim=-1)
                    gt_log_probas.append(prob)
                else:
                    gt_log_probas.append(self.prior(**z_params_i).log_prob(gt_samples[len(z_reps)-1]))
            else:
                prev_zs.append(z_reps[-1])

        # Saving the current latent variable posterior samples in the class attributes and linking them to the
        # back-propagation graph
        if self.inv_seq:
            z_samples = z_samples[::-1]
            z_reps = z_reps[::-1]
            z_log_probas = z_log_probas[::-1]
            z_params = {k: v[::-1] for k, v in z_params.items()}
            if gt_samples is not None:
                gt_log_probas = gt_log_probas[::-1]
        self.post_samples = torch.stack(z_samples, dim=-2)
        self.post_reps = torch.stack(z_reps, dim=-2)
        self.post_log_probas = torch.stack(z_log_probas, dim=-1)
        self.post_params = {k: torch.stack(v, dim=-1-self.prior_params[k].ndim) for k, v in z_params.items()}
        if gt_samples is not None:
            self.post_gt_log_probas = torch.stack(gt_log_probas, dim=-1)
        else:
            self.post_gt_log_probas = None

    def _forward(self, link_approximator, inputs, prior=None, gt_samples=None):
        self.post_params = link_approximator(inputs)
        self.post_samples, self.post_log_probas = self.posterior_sample(self.post_params)
        self.post_reps = self.rep(self.post_samples, step_wise=False)
        if gt_samples is not None:
            if isinstance(self, Gaussian):
                self.post_params['scale_tril'] = torch.diag_embed(self.post_params['scale_tril'])
                self.post_gt_log_probas = self.prior(**self.post_params).log_prob(gt_samples)
            elif isinstance(self, Categorical):
                self.post_params = {**self.post_params, **{'temperature': self.prior_temperature}}
                if gt_samples.dtype == torch.long:
                    gt_samples = F.one_hot(gt_samples, self.size).float()
                    self.post_gt_log_probas = torch.sum(self.post_params['logits'] * gt_samples, dim=-1)
                else:
                    self.post_gt_log_probas = self.prior(**self.post_params).log_prob(gt_samples)
            else:
                raise NotImplementedError("This function still hasn't been implemented for variables other than "
                                          "Gaussians and Categoricals")


# ======================================================================================================================
# ================================================ LATENT VARIABLE CLASSES =============================================

class Gaussian(BaseLatentVariable):
    parameters = {'loc': nn.Sequential(), 'scale_tril': torch.exp}

    def __init__(self, size, name, device, prior_sequential_link=None, posterior=None, markovian=True,
                 allow_prior=False, is_placeholder=False, inv_seq=False, stl=False, repnet=None):
        self.prior_loc = torch.zeros(size).to(device)
        self.prior_cov_tril = torch.eye(size).to(device)
        super(Gaussian, self).__init__(MultivariateNormal, size, {'loc': self.prior_loc,
                                                                  'scale_tril': self.prior_cov_tril},
                                       name, prior_sequential_link, posterior, markovian, allow_prior, is_placeholder,
                                       inv_seq, stl, repnet)

    def infer(self, x_params):
        return x_params['loc']

    def posterior_sample(self, x_params):
        x_params = {**x_params, 'scale_tril': torch.diag_embed(x_params['scale_tril'])}
        return super(Gaussian, self).posterior_sample(x_params)

    def rep(self, samples, step_wise=True, prev_rep=None):
        if self.rep_net is None:
            reps = samples
        else:
            prev_rep = prev_rep.unsqueeze(0).squeeze(-2) if prev_rep is not None else prev_rep
            if step_wise:
                reps = self.rep_net(samples.unsqueeze(-2), hx=prev_rep)[0].squeeze(1)
            else:
                if prev_rep:
                    samples = torch.flip(samples, dims=[-2])
                reps = self.rep_net(samples, hx=prev_rep)[0]
                if prev_rep:
                    reps = torch.flip(reps, dims=[-2])
        return reps


class Categorical(BaseLatentVariable):
    parameters = {'logits': nn.Sequential()}

    def __init__(self, size, name, device, embedding, ignore, prior_sequential_link=None, posterior=None, markovian=True,
                 allow_prior=False, is_placeholder=False, inv_seq=False, stl=False, repnet=None):
        # IDEA: Try to implement "Direct Optimization through argmax"
        self.ignore = ignore
        self.prior_logits = torch.ones(size).to(device)
        self.prior_temperature = torch.tensor([1.0]).to(device)
        super(Categorical, self).__init__(RelaxedOneHotCategorical, size, {'logits': self.prior_logits,
                                                                           'temperature': self.prior_temperature},
                                          name, prior_sequential_link, posterior, markovian, allow_prior,
                                          is_placeholder, inv_seq, stl, repnet)
        self.embedding = embedding
        if self.rep_net is not None:
            embedding_size = embedding.weight.shape[1]
            self.rep_net = repnet or nn.GRU(embedding_size, embedding_size, 1, batch_first=True)

    def infer(self, x_params):
        if 'temperature' not in x_params:
            x_params = {**x_params, **{'temperature': self.prior_temperature}}
        inferred = torch.argmax(x_params['logits'], dim=-1)
        return inferred, self.prior(x_params).log_prob(F.one_hot(inferred, x_params['logits'].shape[-1]))

    def posterior_sample(self, x_params):
        if 'temperature' not in x_params:
            x_params = {**x_params, **{'temperature': self.prior_temperature}}
        #return torch.log(torch.softmax(x_params['logits'], dim=-1)), super(Categorical, self).posterior_sample(x_params)[1]
        return super(Categorical, self).posterior_sample(x_params)

    def rep(self, samples, step_wise=True, prev_rep=None):
        if samples.shape[-1] == self.size and samples.dtype != torch.long:
            embedded = torch.matmul(samples, self.embedding.weight)
        else:
            embedded = self.embedding(samples)
        if self.rep_net is None:
            reps = embedded
        else:
            prev_rep = prev_rep.unsqueeze(0).squeeze(-2) if prev_rep is not None else prev_rep
            if step_wise:
                reps = self.rep_net(embedded.unsqueeze(-2), hx=prev_rep)[0].squeeze(1)
            else:
                if prev_rep:
                    embedded = torch.flip(embedded, dims=[-2])
                reps = self.rep_net(embedded, hx=prev_rep)[0]
                if prev_rep:
                    reps = torch.flip(reps, dims=[-2])
        return reps


class SoftmaxBottleneck(nn.Module):
    def __init__(self, n_experts):
        super(SoftmaxBottleneck, self).__init__()
        self.n_experts = n_experts

    def forward(self, inputs):
        pass



