import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, RelaxedOneHotCategorical, Normal, Independent, OneHotCategorical

from components.links import SequentialLink, NamedLink
from time import time


# ============================================== BASE CLASS ============================================================

class BaseLatentVariable(nn.Module, metaclass=abc.ABCMeta):
    # Define the parameters with their corresponding layer activation function
    # NOTE: identity function behaviour can be produced with torch.nn.Sequential() as an activation function
    parameter_activations = {}

    def __init__(self, prior, size, prior_params, name, prior_sequential_link=None, posterior=None, markovian=True,
                 allow_prior=False, is_placeholder=False, inv_seq=False, stl=False, repnet=None, iw=False,
                 sequence_lv=False, sub_lvl_size=None):
        # IDEA: Lock latent variable behaviour according to it's role in the bayesian network
        super(BaseLatentVariable, self).__init__()
        assert len(self.parameter_activations) > 0
        self.prior_sequential_link = prior_sequential_link
        self.prior = prior
        self.allow_prior = allow_prior
        self.posterior = posterior or prior
        self.size = size
        self.is_placeholder = is_placeholder
        self.inv_seq = inv_seq
        self.stl = stl
        self.iw = iw
        self.sequence_lv = sequence_lv
        self.sub_lvl_size = sub_lvl_size

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
        self.items = self.parameter_activations.items
        self.values = self.parameter_activations.values
        self.keys = self.parameter_activations.keys

        self.post_params = None
        self.post_samples = None
        self.post_reps = None
        self.post_log_probas = None
        self.post_gt_log_probas = None

        self.prev_state = None

    def clear_values(self):
        self.prior_samples = None
        self.prior_reps = None
        self.prior_log_probas = None
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
        try:
            sample = self.posterior(**x_params).rsample()
        except ValueError as e:
            print(self.name)
            print({k: v.shape for k, v in x_params.items()})
            print({k: v for k, v in x_params.items()})
            raise e
        # Applying STL
        if self.stl:
            prior = self.posterior(**{k: v.detach() for k, v in x_params.items()})
        else:
            prior = self.posterior(**{k: v for k, v in x_params.items()})
        return sample, prior.log_prob(sample)

    def prior_sample(self, sample_shape):
        assert not self.inv_seq, "Reversed priors are still not permitted"
        assert self.allow_prior, "{} Doesn't allow for a prior".format(self)
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
                if isinstance(self, Categorical):
                    prior_params_i = {**prior_params_i, **{'temperature': self.prior_temperature}}
            self.prior_samples = torch.stack(self.prior_samples, dim=-2)
            self.prior_log_probas = torch.stack(self.prior_log_probas, dim=-1)

        else:
            prior_distrib = self.prior(**self.prior_params)
            self.prior_samples = prior_distrib.sample(sample_shape)
            self.prior_log_probas = prior_distrib.log_prob(self.prior_samples)
            self.prior_reps = self.rep(self.prior_samples, step_wise=False)

        return self.prior_samples, self.prior_log_probas

    def prior_log_prob(self, sample):
        assert not self.inv_seq, "Reversed priors are still not permitted"
        assert self.allow_prior, "{} Doesn't allow for a prior".format(self)
        if sample.dtype == torch.long and isinstance(self, Categorical):
            sample = F.one_hot(sample, self.size).float()
        if self.prior_sequential_link is not None:
            sample_rep = self.rep(sample, step_wise=False)
            prior_params_i = {k:v.repeat(list(sample.shape[:-2])+[1]*v.ndim) for k, v in self.prior_params.items()}
            prior_log_probas = []
            z_params = {param: [] for param in self.parameter_activations}
            for sample_i, sample_rep_i in zip(sample.transpose(0, -2), sample_rep.transpose(0, -2)):
                for k, v in prior_params_i.items():
                    z_params[k].append(prior_params_i[k])
                prior_distrib = self.prior(**prior_params_i)
                prior_log_probas.append(prior_distrib.log_prob(sample_i))
                prior_params_i = self.prior_sequential_link(sample_rep_i, sample_rep_i)
            for k, v in z_params.items():
                z_params[k] = torch.stack(v, dim=-1-self.prior_params[k].ndim)
            self.post_params = z_params
            return torch.stack(prior_log_probas, dim=-1)

        else:
            prior_distrib = self.prior(**self.prior_params)
            output_params = self.prior_params
            while any([v.ndim < sample.ndim for k, v in output_params.items() if k not in ('temperature', 'n_disc')]):
                output_params = {k: v.unsqueeze(0) for k, v in output_params.items()
                                 if k not in ('temperature', 'n_disc')}
            self.post_params = {k: v.expand((*sample.shape[:-1], self.size)) for k, v in output_params.items()
                                if k not in ('temperature', 'n_disc')}
            return prior_distrib.log_prob(sample)

    def post_log_prob(self, sample):
        if sample.dtype == torch.long and isinstance(self, Categorical):
            assert self.sub_lvl_size is None, "Still to be implemented for sublvl latent variables"
            if sample.shape[-1] != self.size:
                sample = F.one_hot(sample, self.size).float()
            return (self.post_params['logits'].softmax(-1)*sample).sum(-1).log()
        else:
            distrib = self.prior(**self.post_params)
            return distrib.log_prob(sample)

    def forward(self, link_approximator, inputs, prior=None, gt_samples=None, complete=True, lens=None):
        if isinstance(link_approximator, SequentialLink) or (link_approximator.residual is not None and
                                                             isinstance(link_approximator.residual['link'],
                                                                        SequentialLink)):
            self._sequential_forward(link_approximator, inputs, prior, gt_samples)
        else:
            self._forward(link_approximator, inputs, prior, gt_samples, complete=complete, lens=lens)

    @abc.abstractmethod
    def rep(self, samples, step_wise=True, prev_rep=None):
        # Get sample representations
        pass

    def _sequential_forward(self, link_approximator, inputs, prior=None, gt_samples=None):
        if link_approximator.residual is None:
            inputs = torch.cat(list(inputs.values()), dim=-1)
            res_inputs = None
        else:
            non_res_inputs = torch.cat([v for k, v in inputs.items() if k not in link_approximator.residual['conditions']], dim=-1)
            res_inputs = torch.cat([v for k, v in inputs.items() if k in link_approximator.residual['conditions']], dim=-1)
            inputs = non_res_inputs
        if self.inv_seq:
            inputs = torch.flip(inputs, dims=[-2])
            if gt_samples is not None:
                gt_samples = torch.flip(gt_samples, dims=[-2])
            if res_inputs is not None:
                res_inputs = torch.flip(res_inputs, dims=[-2])
        z_params = {param: [] for param in self.prior_params}
        z_log_probas = []
        z_samples = []
        prev_zs = [prior] if prior else []
        if gt_samples is not None:
            prev_zs = self.rep(gt_samples, step_wise=False)
            for i in range(prev_zs.ndim - 2):
                prev_zs = prev_zs.transpose(-3 - i, -2 - i)
            if gt_samples.dtype == torch.long and isinstance(self, Categorical):
                gt_samples = F.one_hot(gt_samples, self.size)
            if gt_samples.dtype == torch.long and isinstance(self, MultiCategorical):
                gt_samples = F.one_hot(gt_samples, int(self.size/self.n_disc))
            for i in range(gt_samples.ndim - 2):
                gt_samples = gt_samples.transpose(-3 - i, -2 - i)
            gt_log_probas = []
        z_reps = []

        x_seq_first = inputs.transpose(-3, -2)
        for i in range(1, x_seq_first.ndim - 2):
            x_seq_first = x_seq_first.transpose(-3 - i, -2 - i)
        if res_inputs is not None:
            x_res_seq_first = res_inputs.transpose(-3, -2)
            for i in range(1, x_res_seq_first.ndim - 2):
                x_res_seq_first = x_res_seq_first.transpose(-3 - i, -2 - i)
        for x in x_seq_first:
            # Inputs are unsqueezed to have single element time-step dimension, while previous outputs are unsqueezed to
            # have a single element num_layers*num_directions dimension (to be changed for multilayer GRUs)
            prev_z = prev_zs[len(z_reps)-1] if len(z_reps) else None
            link_input = (x_res_seq_first[len(z_reps)], x) if res_inputs is not None else x
            z_params_i = link_approximator(link_input, prev_z)
            posterior, posterior_log_prob = self.posterior_sample(z_params_i)
            z_samples.append(posterior)
            z_log_probas.append(posterior_log_prob)
            z_reps.append(self.rep(posterior, prev_rep=prev_z))
            if isinstance(self, Categorical):
                z_params_i = {**z_params_i, **{'temperature': self.prior_temperature}}
            if isinstance(self, MultiCategorical):
                z_params_i = {**z_params_i, **{'temperature': self.prior_temperature, 'n_disc': self.n_disc}}
            for k, v in z_params_i.items():
                z_params[k].append(z_params_i[k])
            if gt_samples is not None:
                if gt_samples.dtype == torch.long:
                    prob = torch.sum(torch.log_softmax(self.post_params['logits'],
                                                       -1) * gt_samples[len(z_reps)-1].float(), dim=-1)
                    gt_log_probas.append(prob)
                else:
                    # Dealing with exploding memory for text probability assessment
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

    def _forward(self, link_approximator, inputs, prior=None, gt_samples=None, complete=True, lens=None):
        if link_approximator.residual is None:
            if not isinstance(link_approximator, NamedLink):
                inputs = torch.cat(list(inputs.values()), dim=-1)
        else:
            assert not isinstance(link_approximator, NamedLink), "can't use a named link in a residual posterior " \
                                                                 "construction (for now)"
            inputs = (torch.cat([v for k, v in inputs.items() if k in link_approximator.residual['conditions']],
                                dim=-1),
                      torch.cat([v for k, v in inputs.items() if k not in link_approximator.residual['conditions']],
                                dim=-1))
        self.post_params = link_approximator(inputs, lens=lens)
        if complete:
            self.post_samples, self.post_log_probas = self.posterior_sample(self.post_params)
            if self.sequence_lv:
                self.post_samples = self.post_samples[..., :1, :].expand(self.post_samples.shape)
                self.post_log_probas = self.post_log_probas[..., :1].expand(self.post_log_probas.shape)
            self.post_reps = self.rep(self.post_samples, step_wise=False)
            if gt_samples is not None:
                if isinstance(self, Gaussian):
                    self.post_gt_log_probas = self.prior(**self.post_params).log_prob(gt_samples)
                elif isinstance(self, Categorical):
                    self.post_params = {**self.post_params, **{'temperature': self.prior_temperature}}
                    if gt_samples.dtype == torch.long:
                        gt_samples = F.one_hot(gt_samples, self.size).float()
                        if self.sub_lvl_size is not None:
                            gt_samples = gt_samples.view(*gt_samples.shape[:-2],
                                                         gt_samples.shape[-2]*gt_samples.shape[-1])
                        self.post_gt_log_probas = torch.sum(torch.log_softmax(self.post_params['logits'],
                                                                              -1) * gt_samples, dim=-1)
                    else:
                        self.post_gt_log_probas = self.prior(**self.post_params).log_prob(gt_samples)
                elif isinstance(self, MultiCategorical):
                    self.post_params = {**self.post_params, **{'temperature': self.prior_temperature,
                                                                'n_disc': self.n_disc}}
                    if gt_samples.dtype == torch.long:
                        gt_samples = F.one_hot(gt_samples, int(self.size/self.n_disc)).float()
                        self.post_gt_log_probas = torch.sum(torch.log_softmax(self.post_params['logits'],
                                                                              -1) * gt_samples, dim=-1)
                    else:
                        self.post_gt_log_probas = self.prior(**self.post_params).log_prob(gt_samples)
                else:
                    raise NotImplementedError("This function still hasn't been implemented for variables other than "
                                              "Gaussians and Categoricals")


# ======================================================================================================================
# ================================================ LATENT VARIABLE CLASSES =============================================

class Gaussian(BaseLatentVariable):
    parameter_activations = {'loc': nn.Sequential(), 'scale': torch.nn.Softplus()}

    def __init__(self, size, name, device, prior_sequential_link=None, posterior=None, markovian=True,
                 allow_prior=False, is_placeholder=False, inv_seq=False, stl=False, repnet=None, iw=False,
                 sequence_lv=False, sub_lvl_size=None):
        self.prior_loc = torch.zeros(size).to(device)
        self.prior_cov = torch.ones(size).to(device)
        super(Gaussian, self).__init__(diag_normal, size, {'loc': self.prior_loc,
                                                           'scale': self.prior_cov},
                                       name, prior_sequential_link, posterior, markovian, allow_prior, is_placeholder,
                                       inv_seq, stl, repnet, iw, sequence_lv, sub_lvl_size=sub_lvl_size)

    def infer(self, x_params):
        return x_params['loc']

    def posterior_sample(self, x_params):
        return super(Gaussian, self).posterior_sample(x_params)

    def rep(self, samples, step_wise=True, prev_rep=None):
        if self.rep_net is None:
            reps = samples
        else:
            depth, directions = self.rep_net.num_layers, 2 if self.rep_net.bidirectional else 1
            prev_rep = prev_rep.view(depth*directions, -1, prev_rep.shape[-1]) if prev_rep is not None else None

            if step_wise:
                flatten = samples.ndim > 2
                if flatten:
                    orig_shape = samples.shape
                    samples = samples.view(-1, orig_shape[-1])
                reps = self.rep_net(samples.unsqueeze(-2), hx=prev_rep)[0].squeeze(1)
            else:
                flatten = samples.ndim > 3
                if flatten:
                    orig_shape = samples.shape
                    samples = samples.reshape(-1, *orig_shape[-2:])
                if self.inv_seq:
                    samples = torch.flip(samples, dims=[-2])
                reps = self.rep_net(samples, hx=prev_rep)[0]
                if self.inv_seq:
                    reps = torch.flip(reps, dims=[-2])
            if flatten:
                reps = reps.view(*orig_shape[:-1], reps.shape[-1])
        return reps

    def get_mi(self, prior_lv):
        # MI upperbound as instructed in "LAGGING INFERENCE NETWORKS AND POSTERIOR
        # COLLAPSE IN VARIATIONAL AUTOENCODERS", He et al. (2019) section 4.2 (originally from Dieng et.al (2018)
        params0, params1 = self.post_params, prior_lv.post_params
        sig0, sig1 = params0['scale'] ** 2, params1['scale'] ** 2

        mu0, mu1 = params0['loc'], params1['loc']

        mean_kl = (0.5 * (sig0 / sig1 + (mu1 - mu0) ** 2 / sig1 + torch.log(sig1) - torch.log(sig0) - 1)).sum(-1).mean()
        log_qz = self.post_log_prob(self.post_samples.unsqueeze(1).expand(-1, self.post_samples.shape[0], -1, -1)).mean(0)
        log_pz = prior_lv.post_log_prob(self.post_samples)
        marginal_kl = (log_qz - log_pz).mean()

        return mean_kl - marginal_kl


class Categorical(BaseLatentVariable):
    parameter_activations = {'logits': nn.Sequential()}

    def __init__(self, size, name, device, embedding, ignore, prior_sequential_link=None, posterior=None, markovian=True,
                 allow_prior=False, is_placeholder=False, inv_seq=False, stl=False, repnet=None, iw=False, sbn_experts=1,
                 word_dropout=None, sequence_lv=False, emb_batch_norm=False, sub_lvl_size=None):
        # IDEA: Try to implement "Direct Optimization through argmax"
        self.ignore = ignore
        self.prior_logits = torch.ones(size).to(device)
        self.prior_temperature = torch.tensor([1.0]).to(device)
        self.sbn_experts = sbn_experts
        super(Categorical, self).__init__(RelaxedOneHotCategorical, size, {'logits': self.prior_logits,
                                                                           'temperature': self.prior_temperature},
                                          name, prior_sequential_link, posterior, markovian, allow_prior,
                                          is_placeholder, inv_seq, stl, repnet, iw, sequence_lv,
                                          sub_lvl_size=sub_lvl_size)
        self.embedding = embedding
        self.w_drp = nn.Dropout(word_dropout) if word_dropout is not None else None
        embedding_size = embedding.weight.shape[1]
        if self.rep_net is not None and not markovian:
            self.rep_net = repnet or nn.GRU(embedding_size, embedding_size, 1, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(embedding_size) if emb_batch_norm else None

    def infer(self, x_params):
        inferred = torch.argmax(x_params['logits'], dim=-1)
        return inferred

    def posterior_sample(self, x_params):
        if 'temperature' not in x_params:
            x_params = {**x_params, **{'temperature': self.prior_temperature}}
        #return torch.log(torch.softmax(x_params['logits'], dim=-1)), super(Categorical, self).posterior_sample(x_params)[1]
        if self.sub_lvl_size is not None:
            logit_shape = x_params['logits'].shape
            x_params['logits'] = x_params['logits'].view(*logit_shape[:-1], self.sub_lvl_size,
                                                         int(logit_shape[-1]/self.sub_lvl_size))
            sample, log_prob = super(Categorical, self).posterior_sample(x_params)
            return sample.view((*sample.shape[:-2], sample.shape[-2]*sample.shape[-1])), log_prob.sum(-1)
        else:
            return super(Categorical, self).posterior_sample(x_params)

    def rep(self, samples, step_wise=True, prev_rep=None):
        if self.sub_lvl_size is not None and samples.dtype != torch.long:
            sample_shape = samples.shape
            samples = samples.view(*sample_shape[:-1], self.sub_lvl_size, int(sample_shape[-1]/self.sub_lvl_size))
        if samples.shape[-1] == self.size and samples.dtype != torch.long:
            embedded = torch.matmul(samples, self.embedding.weight)
        else:
            embedded = self.embedding(samples)
        if self.batch_norm is not None:
            embd_shape = embedded.shape
            embedded = self.batch_norm(embedded.view((-1, embedded.shape[-1]))).view(embd_shape)

        if self.w_drp is not None:
            drp_mask = self.w_drp(torch.ones(embedded.shape[:-1], device=self.prior_logits.device)).unsqueeze(-1)
            embedded = embedded*drp_mask*(1-self.w_drp.p)

        if self.rep_net is None:
            reps = embedded
        else:
            if step_wise:
                depth, directions = self.rep_net.num_layers, 2 if self.rep_net.bidirectional else 1
                if prev_rep is not None and len(prev_rep.shape) != 3:
                    prev_rep = prev_rep.view(depth * directions, -1, prev_rep.shape[-1])
                flatten = embedded.ndim > 2
                if flatten:
                    orig_shape = embedded.shape
                    embedded = embedded.view(-1, orig_shape[-1])
                reps, self.prev_state = self.rep_net(embedded.unsqueeze(-2), hx=prev_rep)
                reps = reps.squeeze(1)
            else:
                flatten = embedded.ndim > 3
                if flatten:
                    orig_shape = embedded.shape
                    embedded = embedded.view(-1, *orig_shape[-2:])
                if self.inv_seq:
                    embedded = torch.flip(embedded, dims=[-2])
                reps, self.prev_state = self.rep_net(embedded, hx=prev_rep)
                if self.inv_seq:
                    reps = torch.flip(reps, dims=[-2])
            if flatten:
                reps = reps.view(*orig_shape[:-1], reps.shape[-1])
        if self.sub_lvl_size is not None:
            reps = reps.view((*reps.shape[:-2], reps.shape[-2]*reps.shape[-1]))
        return reps

    def switch_to_relaxed(self):
        self.prior = self.posterior = RelaxedOneHotCategorical

    def switch_to_non_relaxed(self):
        self.prior = self.posterior = MyOneHotCategorical


class MultiCategorical(BaseLatentVariable):
    parameter_activations = {'logits': nn.Sequential()}

    def __init__(self, size, name, device, embedding, ignore, prior_sequential_link=None, posterior=None, markovian=True,
                 allow_prior=False, is_placeholder=False, inv_seq=False, stl=False, repnet=None, iw=False,
                 word_dropout=None, sequence_lv=False, n_disc=1, sub_lvl_size=None):
        # IDEA: Try to implement "Direct Optimization through argmax"
        self.ignore = ignore
        assert size % n_disc == 0
        self.n_disc = torch.tensor([n_disc]).to(device)
        torch.rand(size, device=device, requires_grad=True)
        self.prior_logits = torch.ones(size).to(device)*5
        self.prior_temperature = torch.tensor([1.0]).to(device)
        super(MultiCategorical, self).__init__(MultiRelaxedOneHotCategorical, size,
                                               {'logits': self.prior_logits, 'temperature': self.prior_temperature,
                                                'n_disc': self.n_disc},
                                               name, prior_sequential_link, posterior, markovian, allow_prior,
                                               is_placeholder, inv_seq, stl, repnet, iw, sequence_lv,
                                               sub_lvl_size=sub_lvl_size)
        assert isinstance(embedding, MultiEmbedding)
        self.embedding = embedding
        self.w_drp = nn.Dropout(word_dropout) if word_dropout is not None else None
        if self.rep_net is not None and not markovian:
            embedding_size = embedding.embeddings[0].weight.shape[1]*n_disc
            self.rep_net = repnet or nn.GRU(embedding_size, embedding_size, 1, batch_first=True)

    def infer(self, x_params):
        # Not sure this works with Relaxed distributions. But it's still not used anyway
        if 'temperature' not in x_params:
            x_params = {**x_params, **{'temperature': self.prior_temperature}, 'n_disc':self.n_disc}
        inferred = torch.argmax(x_params['logits'], dim=-1)
        return inferred

    def posterior_sample(self, x_params):
        if 'temperature' not in x_params:
            x_params = {**x_params, **{'temperature': self.prior_temperature}, 'n_disc':self.n_disc}
        return super(MultiCategorical, self).posterior_sample(x_params)

    def rep(self, samples, step_wise=True, prev_rep=None):
        embedded = self.embedding(samples)

        if self.w_drp is not None:
            drp_mask = self.w_drp(torch.ones(embedded.shape[:-1], device=self.prior_logits.device)).unsqueeze(-1)
            embedded = embedded*drp_mask*(1-self.w_drp.p)

        if self.rep_net is None:
            reps = embedded
        else:
            if step_wise:
                depth, directions = self.rep_net.num_layers, 2 if self.rep_net.bidirectional else 1
                if prev_rep is not None and len(prev_rep.shape) != 3:
                    prev_rep = prev_rep.view(depth * directions, -1, prev_rep.shape[-1])
                flatten = embedded.ndim > 2
                if flatten:
                    orig_shape = embedded.shape
                    embedded = embedded.view(-1, orig_shape[-1])
                reps, self.prev_state = self.rep_net(embedded.unsqueeze(-2), hx=prev_rep)
                reps = reps.squeeze(1)
            else:
                flatten = embedded.ndim > 3
                if flatten:
                    orig_shape = embedded.shape
                    embedded = embedded.view(-1, *orig_shape[-2:])
                if self.inv_seq:
                    embedded = torch.flip(embedded, dims=[-2])
                reps, self.prev_state = self.rep_net(embedded, hx=prev_rep)
                if self.inv_seq:
                    reps = torch.flip(reps, dims=[-2])
            if flatten:
                reps = reps.view(*orig_shape[:-1], reps.shape[-1])
        return reps

    def switch_to_relaxed(self):
        self.prior = self.posterior = MultiRelaxedOneHotCategorical

    def switch_to_non_relaxed(self):
        self.prior = self.posterior = MultiOneHotCategorical


def diag_normal(loc, scale):
    return Independent(Normal(loc, scale), 1)


class SoftmaxBottleneck(nn.Module):
    def __init__(self, n_experts=4):
        super(SoftmaxBottleneck, self).__init__()
        self.n_experts = n_experts
        self.exp_weights = torch.nn.Parameter(torch.randn(n_experts), requires_grad=True)

    def forward(self, inputs, weights):
        assert weights.shape[1] % self.n_experts == 0, "logits size {} is indivisible by numbert of " \
                                                       "experts".format(weights.shape[1], self.n_experts)
        weights = weights.transpose(0, 1).reshape(self.n_experts, int(weights.shape[1]/self.n_experts), weights.shape[0])
        experts = []
        for weight, exp_w in zip(weights, self.exp_weights):
            experts.append(torch.matmul(inputs, weight) * exp_w)
        outputs = torch.sum(torch.softmax(torch.stack(experts), dim=-1), dim=0)/torch.sum(self.exp_weights)

        return torch.log(outputs+1e-8)


class MultiEmbedding(nn.Module):
    def __init__(self, size, n_disc, dim):
        super(MultiEmbedding, self).__init__()
        assert size % n_disc == 0
        assert dim % n_disc == 0
        self.n_disc = n_disc
        self.dim = dim
        self.embeddings = nn.ModuleList([nn.Embedding(int(size/n_disc), int(dim/n_disc)) for _ in range(n_disc)])

    def forward(self, inputs):
        assert inputs.shape[-1] % self.n_disc == 0
        inputs = inputs.reshape(inputs.shape[:-1]+(self.n_disc, int(inputs.shape[-1]/self.n_disc)))
        if inputs.dtype == torch.long:
            outputs = torch.cat([emb[inputs_i] for emb, inputs_i in zip(self.embeddings, inputs[..., :],)], -1)
        else:
            outputs = torch.cat([torch.matmul(inputs[..., i, :], emb.weight) for emb, i in zip(self.embeddings,
                                                                   range(self.n_disc))], -1)
        return outputs


class MultiRelaxedOneHotCategorical(RelaxedOneHotCategorical):
    def __init__(self, temperature, n_disc, probs=None, logits=None, validate_args=None):
        probs = None if probs is None else probs.reshape(probs.shape[:-1]+(n_disc, int(probs.shape[-1]/n_disc)))
        logits = None if logits is None else logits.reshape(logits.shape[:-1]+(n_disc, int(logits.shape[-1]/n_disc)))
        self.n_disc = n_disc
        super(MultiRelaxedOneHotCategorical, self).__init__(temperature, probs=probs, logits=logits,
                                                            validate_args=validate_args)

    def log_prob(self, value):
        value = value.reshape(value.shape[:-1]+(self.n_disc, int(value.shape[-1]/self.n_disc)))
        disc_wise_log_prob = super(MultiRelaxedOneHotCategorical, self).log_prob(value)
        return disc_wise_log_prob.sum(-1)

    def rsample(self, sample_shape=torch.Size()):
        sample = super(MultiRelaxedOneHotCategorical, self).rsample(sample_shape)
        return sample.reshape(sample.shape[:-2]+(sample.shape[-1]*sample.shape[-2],))

    def sample(self, sample_shape=torch.Size()):
        sample = super(MultiRelaxedOneHotCategorical, self).sample(sample_shape)
        return sample.reshape(sample.shape[:-2]+(sample.shape[-1]*sample.shape[-2],))


class MultiOneHotCategorical(OneHotCategorical):
    def __init__(self, temperature, n_disc, probs=None, logits=None, validate_args=None):
        probs = None if probs is None else probs.reshape(probs.shape[:-1]+(n_disc, int(probs.shape[-1]/n_disc)))
        logits = None if logits is None else logits.reshape(logits.shape[:-1]+(n_disc, int(logits.shape[-1]/n_disc)))
        self.n_disc = n_disc
        super(MultiOneHotCategorical, self).__init__(probs=probs, logits=logits,
                                                     validate_args=validate_args)

    def log_prob(self, value):
        value = value.reshape(value.shape[:-1]+(self.n_disc, int(value.shape[-1]/self.n_disc)))
        disc_wise_log_prob = super(MultiOneHotCategorical, self).log_prob(value)
        return disc_wise_log_prob.sum(-1)

    def rsample(self, sample_shape=torch.Size()):
        sample = super(MultiOneHotCategorical, self).sample(sample_shape)
        return sample.reshape(sample.shape[:-2]+(sample.shape[-1]*sample.shape[-2],))

    def sample(self, sample_shape=torch.Size()):
        sample = super(MultiOneHotCategorical, self).sample(sample_shape)
        return sample.reshape(sample.shape[:-2]+(sample.shape[-1]*sample.shape[-2],))



class MyOneHotCategorical(OneHotCategorical):
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super(MyOneHotCategorical, self).__init__(probs=probs, logits=logits, validate_args=validate_args)


