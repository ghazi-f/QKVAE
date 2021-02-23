import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.latent_variables import Categorical, Gaussian, MultiCategorical

from time import time


# ============================================== BASE CLASS ============================================================

class BaseCriterion(metaclass=abc.ABCMeta):
    def __init__(self, model, w):
        self.model = model
        self.h_params = model.h_params
        self.w = w
        self._prepared_metrics = None

    @abc.abstractmethod
    def get_loss(self):
        # The loss function
        pass

    def metrics(self):
        return self._prepared_metrics

    @abc.abstractmethod
    def _prepare_metrics(self, loss):
        pass


# ============================================== CRITERIA CLASSES ======================================================

class Supervision(BaseCriterion):
    def __init__(self, model, w):
        # Warning: This is still only implemented for categorical supervised variables
        super(Supervision, self).__init__(model, w)
        self.supervised_lv = model.supervised_v
        self.net = model.infer_bn

        criterion_params = {'ignore_index': self.supervised_lv.ignore}
        if self.supervised_lv.name in self.h_params.is_weighted:
            counts = [model.index[self.supervised_lv].freqs[w] for w in self.model.index[self.supervised_lv].itos]
            freqs = torch.sqrt(torch.Tensor([1/c if c != 0 else 0 for c in counts]).to(self.h_params.device))
            criterion_params['weight'] = freqs/torch.sum(freqs)*len(freqs)

        if isinstance(self.supervised_lv, Categorical):
            self.criterion = nn.CrossEntropyLoss(**criterion_params)
        else:
            raise NotImplementedError('The supervision criterium has not been implemented yet '
                                      'for {} latent variables'.format(self.supervised_lv.name))

    def get_loss(self):
        num_classes = self.supervised_lv.size
        predictions = self.supervised_lv.post_params['logits']
        target = self.net.variables_star[self.supervised_lv]
        # Taking the first sample in case of importance sampling
        while predictions.ndim-target.ndim > 1:
            predictions = predictions[0]
        predictions = predictions.reshape(-1, num_classes)
        target = target.reshape(-1)
        loss = self.criterion(predictions, target)

        self._prepare_metrics(loss)

        # Trying to anneal this loss too
        # if self.h_params.anneal_kl:
        #     anl0, anl1 = self.h_params.anneal_kl[0], self.h_params.anneal_kl[1]
        #     coeff = 0 if self.model.step < anl0 else ((self.model.step-anl0)/(anl1 - anl0)) if anl1 > self.model.step >= anl0 else 1
        #     coeff = torch.tensor(coeff)
        # else:
        #     coeff = torch.tensor(1)

        return loss

    def _prepare_metrics(self, loss):
        ce = loss
        with torch.no_grad():
            num_classes = self.supervised_lv.size
            predictions = self.supervised_lv.post_params['logits']
            target = self.net.variables_star[self.supervised_lv]
            # Taking the first sample in case of importance sampling
            while predictions.ndim-target.ndim > 1:
                predictions = predictions[0]
            predictions = predictions.reshape(-1, num_classes)
            target = target.reshape(-1)
            prediction_mask = (target != self.supervised_lv.ignore).float()
            accuracy = torch.sum((torch.argmax(predictions, dim=-1) == target).float()*prediction_mask)
            accuracy /= torch.sum(prediction_mask)
        self._prepared_metrics = {'/{}_CE'.format(self.supervised_lv.name): ce,
                                  '/{}_accuracy'.format(self.supervised_lv.name): accuracy}


class ELBo(BaseCriterion):
    # This is actually Sticking The Landing (STL) ELBo, and not the standard one
    # (Same gradient expectancy but less gradient variance)
    def __init__(self, model, w):
        super(ELBo, self).__init__(model, w)
        self.infer_net = model.infer_bn
        self.gen_net = model.gen_bn

        # Taking the variable that has no children as the target for generation
        self.generated_v = model.generated_v
        self.infer_lvs = {lv.name: lv for lv in self.infer_net.variables if lv.name != self.generated_v.name
                          and not lv.is_placeholder}
        self.gen_lvs = {lv.name: lv for lv in self.gen_net.variables if lv.name != self.generated_v.name
                        and not lv.is_placeholder}

        # Warning: This is still only implemented for categorical generation
        criterion_params = {'ignore_index': self.generated_v.ignore, 'reduction': 'none'}
        if self.generated_v.name in self.h_params.is_weighted:
            counts = [model.index[self.generated_v].freqs[w] for w in self.model.index[self.generated_v].itos]
            freqs = torch.sqrt(torch.Tensor([1/c if c != 0 else 0 for c in counts]).to(self.h_params.device))
            criterion_params['weight'] = freqs/torch.sum(freqs)*len(freqs)
        self.criterion = nn.CrossEntropyLoss(**criterion_params)
        if self.generated_v.name in self.h_params.is_weighted:
            criterion_params.pop('weight')
        self._unweighted_criterion = nn.CrossEntropyLoss(**criterion_params)

        self.log_p_xIz = None
        self.log_p_z = None
        self.log_q_zIx = None

        self.sequence_mask = None
        self.valid_n_samples = None

    def get_loss(self, actual=False, observed=None):
        vocab_size = self.generated_v.size
        criterion = self._unweighted_criterion if actual else self.criterion
        self.sequence_mask = (self.gen_net.variables_star[self.generated_v] != self.generated_v.ignore).float()
        temp = 1

        self.log_p_xIz = - criterion(self.generated_v.post_params['logits'].view(-1, vocab_size)/temp,
                                     self.gen_net.variables_star[self.generated_v].reshape(-1)
                                     ).view(self.gen_net.variables_star[self.generated_v].shape) * self.sequence_mask
        if self.generated_v.sub_lvl_size is not None:
            self.sequence_mask = self.sequence_mask.sum(-1)
            self.log_p_xIz = self.log_p_xIz.sum(-1)

        self.valid_n_samples = torch.sum(self.sequence_mask)
        # Applying KL Thresholding (Free Bits)
        if self.h_params.kl_th is None or actual:
            thr = None
        else:
            thr = torch.tensor([self.h_params.kl_th]).to(self.h_params.device)
        kl = sum([kullback_liebler(self.infer_lvs[lv_n], self.gen_lvs[lv_n], thr=thr)
                  for lv_n in self.infer_lvs.keys() if observed is None or (lv_n not in observed)])
        if observed is not None:
            self.log_p_xIz += sum([self.gen_net.log_proba[lv] for lv in self.gen_lvs.values() if lv.name in observed]) \
                              * self.sequence_mask
        kl *= self.sequence_mask

        # Applying KL Annealing
        if self.h_params.anneal_kl and not actual:
            anl0, anl1 = self.h_params.anneal_kl[0], self.h_params.anneal_kl[1]
            is_sigmoid = anl0==-1
            if is_sigmoid:
                pass
            else:
                coeff = 0 if self.model.step < anl0 else ((self.model.step-anl0)/(anl1 - anl0)) if anl1 > self.model.step >= anl0 else 1
            coeff = torch.tensor(coeff)
        else:
            coeff = torch.tensor(1)
        if coeff == 0:
            kl = 0
        if self.h_params.max_elbo:
            # Didn't implement observed here
            loss_choice, loss_threshold = self.h_params.max_elbo
            if not actual and type(kl) != int:
                if loss_choice == 0:
                    max_kl = torch.max(torch.stack([kullback_liebler(self.infer_lvs[lv_n], self.gen_lvs[lv_n], thr=thr)
                              for lv_n in self.infer_lvs.keys()]), dim=0)[0]
                    loss = - torch.sum(torch.min(self.log_p_xIz, -coeff * max_kl/loss_threshold)
                                       , dim=(0, 1)) / self.valid_n_samples
                if loss_choice == 1:
                    kl_stack = torch.stack([kullback_liebler(self.infer_lvs[lv_n], self.gen_lvs[lv_n], thr=thr)
                              for lv_n in self.infer_lvs.keys()])
                    max_max_kl = kl_stack.max(0)[0]
                    max_kl = (kl_stack-max_max_kl.unsqueeze(0)).exp().sum(0).log()+max_max_kl*kl_stack.shape[0]
                    loss = torch.sum((-self.log_p_xIz.exp()+(coeff * max_kl/loss_threshold).exp()).log(),
                                     dim=(0, 1)) / self.valid_n_samples
                elif loss_choice == 2:
                    anl0, anl1 = self.h_params.anneal_kl[0], self.h_params.anneal_kl[1]
                    zi_coeffs = {'z'+str(len(self.h_params.n_latents)+1-i): 0 if self.model.step < anl0 else (
                                        (self.model.step - anl0) / ((anl1 - anl0)*i))
                                 if (anl1 - anl0)*i+anl0 > self.model.step >= anl0 else 1
                                 for i in range(1, len(self.h_params.n_latents)+1)
                                 }
                    this_kl = sum([kullback_liebler(self.infer_lvs[lv_n], self.gen_lvs[lv_n], thr=thr)*zi_coeffs[lv_n]
                                   for lv_n in self.infer_lvs.keys()
                                   if observed is None or (lv_n not in observed)])
                    this_kl *= self.sequence_mask
                    loss = - torch.sum(torch.min(self.log_p_xIz, - this_kl/loss_threshold),
                                       dim=(0, 1)) / self.valid_n_samples
                elif loss_choice == 3:
                    anl0, anl1 = self.h_params.anneal_kl[0], self.h_params.anneal_kl[1]
                    n_lat = self.h_params.n_latents
                    zi_coeffs = {'z'+str(len(n_lat)+1-i): (0 if self.model.step < anl0 else (
                                        (self.model.step - anl0) / ((anl1 - anl0)*i))
                                 if (anl1 - anl0)*i+anl0 > self.model.step >= anl0 else 1) * max(n_lat) / n_lat[-i]
                                 for i in range(1, len(self.h_params.n_latents)+1)
                                 }
                    this_kl = torch.max(torch.stack([kullback_liebler(self.infer_lvs[lv_n], self.gen_lvs[lv_n], thr=thr)
                                                     * zi_coeffs[lv_n]
                                                     for lv_n in self.infer_lvs.keys()]), dim=0)[0]
                    this_kl *= self.sequence_mask
                    loss = - torch.sum(torch.min(self.log_p_xIz, - this_kl/loss_threshold),
                                       dim=(0, 1)) / self.valid_n_samples
                elif loss_choice == 4:

                    anl0, anl1 = self.h_params.anneal_kl[0], self.h_params.anneal_kl[1]
                    anl_gap = anl1-anl0
                    zi_coeffs = {'z'+str(len(self.h_params.n_latents)+1-i): 0 if self.model.step < anl0+anl_gap*(i-1)
                                 else ((self.model.step - (anl0+anl_gap*(i-1))) / anl_gap)
                                 if anl_gap*i+anl0 > self.model.step >= anl0+anl_gap*(i-1) else 1
                                 for i in range(1, len(self.h_params.n_latents)+1)
                                 }
                    this_kl = sum([kullback_liebler(self.infer_lvs[lv_n], self.gen_lvs[lv_n], thr=thr)*zi_coeffs[lv_n]
                                   for lv_n in self.infer_lvs.keys()
                                   if observed is None or (lv_n not in observed)])
                    this_kl *= self.sequence_mask
                    loss = - torch.sum(torch.min(self.log_p_xIz, - this_kl/loss_threshold),
                                       dim=(0, 1)) / self.valid_n_samples

                # loss = - torch.sum(torch.min(self.log_p_xIz, -coeff * this_kl/loss_threshold),
                #                    dim=(0, 1)) / self.valid_n_samples
            else:
                loss = - torch.sum(self.log_p_xIz - coeff * kl, dim=(0, 1))/self.valid_n_samples
        else:
            loss = - torch.sum(self.log_p_xIz - coeff * kl, dim=(0, 1))/self.valid_n_samples

        with torch.no_grad():
            if observed is None:
                if actual and thr is None:
                    unweighted_loss = loss
                else:
                    un_log_p_xIz = \
                        - self._unweighted_criterion(self.generated_v.post_params['logits'].view(-1, vocab_size)/temp,
                                                     self.gen_net.variables_star[self.generated_v].reshape(-1)
                                                     ).view(self.gen_net.variables_star[self.generated_v].shape)
                    if self.generated_v.sub_lvl_size is not None:
                        un_log_p_xIz = un_log_p_xIz.sum(-1)
                    un_log_p_xIz *= self.sequence_mask
                    kl = sum([kullback_liebler(self.infer_lvs[lv_n], self.gen_lvs[lv_n], thr=None)
                              for lv_n in self.infer_lvs.keys()]) * self.sequence_mask
                    unweighted_loss = - torch.sum(un_log_p_xIz - kl, dim=(0, 1))/self.valid_n_samples
                self._prepare_metrics(unweighted_loss)

        return loss

    def _prepare_metrics(self, loss):
        current_elbo = - loss
        LL_name = '/p({}I{})'.format(self.generated_v.name, ', '.join([lv for lv in sorted(self.infer_lvs.keys())]))
        LL_value = torch.sum(self.log_p_xIz)/self.valid_n_samples
        KL_dict = {}
        for lv in self.gen_lvs.keys():
            if lv not in self.infer_lvs: continue
            gen_lv, inf_lv = self.gen_lvs[lv], self.infer_lvs[lv]
            infer_v_name = inf_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.infer_net.parent[inf_lv]]))
                                          if inf_lv in self.infer_net.parent else '')
            gen_v_name = gen_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.gen_net.parent[gen_lv]]))
                                        if gen_lv in self.gen_net.parent else '')
            KL_name = '/KL(q({})IIp({}))'.format(infer_v_name, gen_v_name)
            kl_i = kullback_liebler(inf_lv, gen_lv)*self.sequence_mask
            KL_value = torch.sum(kl_i)/self.valid_n_samples
            KL_dict[KL_name] = KL_value
        if not (type(self.h_params.n_latents) == int and self.h_params.n_latents == 1):
            for j, name in enumerate(self.infer_lvs.keys()):
                if name in self.gen_lvs:
                    gen_lv, inf_lv = self.gen_lvs[name], self.infer_lvs[name]
                    infer_v_name = inf_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.infer_net.parent[inf_lv]]))
                                                  if inf_lv in self.infer_net.parent else '')
                    gen_v_name = gen_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.gen_net.parent[gen_lv]]))
                                                if gen_lv in self.gen_net.parent else '')
                    KLs = []
                    KL_var_name = '/VarKL(q({})IIp({}))'.format(infer_v_name,gen_v_name)
                    n_latents= self.h_params.n_latents[j]
                    for i in range(n_latents):
                        if gen_lv.post_params is None: continue
                        start, end = int(i*self.h_params.z_size/n_latents), \
                                     int((i+1)*self.h_params.z_size/n_latents)
                        kl_i = kullback_liebler(inf_lv, gen_lv, slice=(start, end)) * self.sequence_mask
                        KLs.append(torch.sum(kl_i) / self.valid_n_samples)
                    KL_dict[KL_var_name] = torch.std(torch.tensor(KLs))

        self._prepared_metrics = {'/ELBo': current_elbo, LL_name: LL_value, **KL_dict}


class Reconstruction(BaseCriterion):
    def __init__(self, model, w):
        super(Reconstruction, self).__init__(model, w)
        self.gen_net = model.gen_bn

        # Taking the variable that has no children as the target for generation
        self.generated_v = model.generated_v

        # Warning: This is still only implemented for categorical generation
        criterion_params = {'ignore_index': self.generated_v.ignore, 'reduction': 'mean'}
        if self.generated_v.name in self.h_params.is_weighted:
            counts = [model.index[self.generated_v].freqs[w] for w in self.model.index[self.generated_v].itos]
            freqs = torch.sqrt(torch.Tensor([1/c if c != 0 else 0 for c in counts]).to(self.h_params.device))
            criterion_params['weight'] = freqs/torch.sum(freqs)*len(freqs)
        self.criterion = nn.CrossEntropyLoss(**criterion_params)
        if self.generated_v.name in self.h_params.is_weighted:
            criterion_params.pop('weight')
        self._unweighted_criterion = nn.CrossEntropyLoss(**criterion_params)

        self.log_p_x = None

        self.sequence_mask = None
        self.valid_n_samples = None

    def get_loss(self, actual=False):

        vocab_size = self.generated_v.size
        criterion = self._unweighted_criterion if actual else self.criterion
        temp = 1
        self.log_p_x = - criterion(self.generated_v.post_params['logits'].view(-1, vocab_size)/temp,
                                     self.gen_net.variables_star[self.generated_v].reshape(-1))
        loss = - self.log_p_x

        with torch.no_grad():
            if actual:
                self._prepare_metrics(loss)
            else:
                un_log_p_x = - self._unweighted_criterion(self.generated_v.post_params['logits'].view(-1, vocab_size)/temp,
                                                          self.gen_net.variables_star[self.generated_v].reshape(-1)
                                                          )
                self._prepare_metrics(un_log_p_x)

        return loss

    def _prepare_metrics(self, loss):
        current_ll = - loss
        LL_name = '/p({})'.format(self.generated_v.name)

        self._prepared_metrics = {LL_name: current_ll}


class IWLBo(ELBo):
    # This is actually Doubly Reparameterized Gradient (DReG) IWLBo and not the standard one
    # (Same gradient expectancy but less gradient variance)
    def __init__(self, model, w):
        super(IWLBo, self).__init__(model, w)
        self.input_dimensions = self.h_params.input_dimensions

    def get_loss(self, actual=False, observed=None):

        vocab_size = self.generated_v.size
        criterion = self._unweighted_criterion if actual else self.criterion
        self.sequence_mask = (self.gen_net.variables_star[self.generated_v] != self.generated_v.ignore).float()
        if self.generated_v.sub_lvl_size is not None:
            self.sequence_mask = self.sequence_mask.sum(-1)
        self.valid_n_samples = torch.sum(self.sequence_mask)
        loss_shape = self.generated_v.post_params['logits'].shape[:-1]
        if len(loss_shape) > 2:
            logits, gt = self.generated_v.post_params['logits'], self.gen_net.variables_star[self.generated_v]
            if self.generated_v.sub_lvl_size is not None:
                batchxseq_size = gt.shape[-3]*gt.shape[-2]*gt.shape[-1]
            else:
                batchxseq_size = gt.shape[-2]*gt.shape[-1]
            logits, gt = logits.reshape(-1, batchxseq_size, vocab_size), torch.unbind(gt.reshape(-1, batchxseq_size))[0]
            log_p_xIz = []
            for logits_i in torch.unbind(logits):
                log_p_xIz.append(- criterion(logits_i, gt))
            if self.generated_v.sub_lvl_size:
                log_p_xIz = torch.cat(log_p_xIz, dim=0).view((*loss_shape, self.generated_v.sub_lvl_size)).sum(-1)
            else:
                log_p_xIz = torch.cat(log_p_xIz, dim=0).view(loss_shape)
        else:
            log_p_xIz = - criterion(self.generated_v.post_params['logits'].view(-1, vocab_size),
                                    self.gen_net.variables_star[self.generated_v].reshape(-1)
                                    ).view(self.generated_v.post_params['logits'].shape[:-1])

        # Applying KL Annealing (or it's equivalent for IWAEs)
        if self.h_params.anneal_kl and not actual:
            anl0, anl1 = self.h_params.anneal_kl[0], self.h_params.anneal_kl[1]
            coeff = 0 if self.model.step < anl0 else ((self.model.step-anl0)/(anl1 - anl0)) if anl1 > self.model.step >= anl0 else 1
            coeff = torch.tensor(coeff)
        else:
            coeff = torch.tensor(1)
        if coeff == 0:
            log_p_z = 0
            log_q_zIx = 0
        else:
            log_p_z = [self.gen_net.log_proba[lv] for lv in self.gen_lvs.values()
                       if observed is None or (lv.name not in observed)]
            log_q_zIx = [self.infer_net.log_proba[lv] for lv in self.infer_lvs.values()
                         if observed is None or (lv.name not in observed)]

            # Filling in for additional dimensions in shapes when it's needed

            max_dims = log_p_z[0].ndim
            for i in range(len(log_q_zIx)):
                dims_i = log_q_zIx[i].ndim
                for _ in range(max_dims-dims_i):
                    log_q_zIx[i] = log_q_zIx[i].unsqueeze(0)

            # Applying sequence mask to all log probabilities
            log_p_z = sum(log_p_z) * self.sequence_mask
            log_q_zIx = sum(log_q_zIx) * self.sequence_mask
        log_p_xIz_obs = sum([self.gen_net.log_proba[lv] for lv in self.gen_lvs.values()
                         if (lv.name in observed)]) if observed is not None else 0
        log_p_xIz = (log_p_xIz + log_p_xIz_obs) * self.sequence_mask

        # Calculating IWLBo Gradient estimate
        log_wi = coeff * (log_p_z - log_q_zIx) + log_p_xIz
        detached_log_wi = log_wi.detach()
        max_log_wi = torch.max(detached_log_wi)
        detached_exp_log_wi = torch.exp(detached_log_wi - max_log_wi)

        if actual:
            n_samples_correction = 1
            while detached_exp_log_wi.ndim > self.input_dimensions:
                n_samples_correction *= detached_exp_log_wi.shape[0]
                detached_exp_log_wi = torch.mean(detached_exp_log_wi, dim=0)
            loss = - torch.sum(torch.log(detached_exp_log_wi+1e-8) +
                               max_log_wi)/(self.valid_n_samples/n_samples_correction)
        else:
            DReG_weights = (detached_exp_log_wi / (1e-8 + torch.sum(detached_exp_log_wi, dim=0).unsqueeze(0)))**2
            loss = - torch.sum(DReG_weights * log_wi)/self.valid_n_samples

        with torch.no_grad():
            if observed is None:
                if actual:
                    unweighted_loss = loss
                else:
                    log_wi = (log_p_z - log_q_zIx) + log_p_xIz
                    max_log_wi = torch.max(log_wi)
                    exp_log_wi = torch.exp(log_wi - max_log_wi)
                    n_samples_correction = 1
                    while exp_log_wi.ndim > self.input_dimensions:
                        n_samples_correction *= exp_log_wi.shape[0]
                        exp_log_wi = torch.mean(exp_log_wi, dim=0)
                    summed_log_wi = torch.log(exp_log_wi+1e-8) + max_log_wi
                    unweighted_loss = - torch.sum(summed_log_wi)/(self.valid_n_samples/n_samples_correction)
                self.ll_value = torch.sum(log_p_xIz) / self.valid_n_samples
                self._prepare_metrics(unweighted_loss)

        return loss

    def _prepare_metrics(self, loss):
        current_iwlbo = - loss
        LL_name = '/p({}I{})'.format(self.generated_v.name, ', '.join([lv for lv in self.infer_lvs]))
        LL_value = self.ll_value
        KL_dict = {}
        kl_sum = 0
        if self.model.step >= self.h_params.anneal_kl[0]:
            for lv in self.gen_lvs.keys():
                if lv not in self.infer_lvs: continue
                gen_lv, inf_lv = self.gen_lvs[lv], self.infer_lvs[lv]
                infer_v_name = inf_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.infer_net.parent[inf_lv]]))
                                              if inf_lv in self.infer_net.parent else '')
                gen_v_name = gen_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.gen_net.parent[gen_lv]]))
                                            if gen_lv in self.gen_net.parent else '')
                KL_name = '/KL(q({})IIp({}))'.format(infer_v_name, gen_v_name)
                kl_i = kullback_liebler(inf_lv, gen_lv)*self.sequence_mask
                KL_value = torch.sum(kl_i)/self.valid_n_samples
                kl_sum += KL_value
                KL_dict[KL_name] = KL_value

        self._prepared_metrics = {'/IWLBo': current_iwlbo, '/ELBo': LL_value-kl_sum, LL_name: LL_value, **KL_dict}


def kullback_liebler(lv0, lv1, thr=None, slice=None):
    # Accounting for the case when it's not estimated do to pure reconstruction phase
    params0, params1 = lv0.post_params, lv1.post_params
    assert lv0.sub_lvl_size is None and lv1.sub_lvl_size is None \
        , "Kullback leibler for sublvl variables is still not implemented"
    if params1 is None:
        return 0
    if isinstance(lv0, Gaussian) and isinstance(lv1, Gaussian):
        # The gaussian case
        sig0, sig1 = params0['scale']**2, params1['scale']**2

        mu0, mu1 = params0['loc'], params1['loc']

        kl_per_dim = 0.5*(sig0/sig1+(mu1-mu0)**2/sig1 + torch.log(sig1) - torch.log(sig0) - 1)
        if slice is not None:
            kl_per_dim = kl_per_dim[..., slice[0]:slice[1]]
        if thr is not None:
            kl_per_dim = torch.max(kl_per_dim, thr)
        return torch.sum(kl_per_dim, dim=-1)
    elif isinstance(lv0, Categorical) and isinstance(lv1, Categorical):
        assert slice is None
        # The categorical case
        logit0, logit1 = params0['logits'], params1['logits']
        kl_per_dim = torch.softmax(logit0, dim=-1)*(torch.log_softmax(logit0, dim=-1) -
                                                    torch.log_softmax(logit1, dim=-1))
        if thr is not None:
            kl_per_dim = torch.max(kl_per_dim, thr)
        return torch.sum(kl_per_dim, dim=-1)
    elif isinstance(lv0, MultiCategorical) and isinstance(lv1, MultiCategorical):
        # The multicategorical case
        logit0, logit1 = params0['logits'], params1['logits']
        logit0 = logit0.reshape(logit0.shape[:-1]+(lv0.n_disc, int(logit0.shape[-1]/lv0.n_disc)))
        logit1 = logit1.reshape(logit1.shape[:-1]+(lv1.n_disc, int(logit1.shape[-1]/lv1.n_disc)))
        kl_per_dim = torch.softmax(logit0, dim=-1)*(torch.log_softmax(logit0, dim=-1) -
                                                    torch.log_softmax(logit1, dim=-1))
        kl_per_dim = kl_per_dim.reshape(kl_per_dim.shape[:-2]+(kl_per_dim.shape[-2]*kl_per_dim.shape[-1],))
        if slice is not None:
            kl_per_dim = kl_per_dim[..., slice[0]:slice[1]]
        if thr is not None:
            kl_per_dim = torch.max(kl_per_dim, thr)
        return torch.sum(kl_per_dim, dim=-1)
    else:
        raise NotImplementedError('The cas where lv0 is {} and lv1 is {} '
                                  'is not implemented yet'.format(repr(type(lv0)), repr(type(lv0))))
