import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.latent_variables import Categorical


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
            freqs = torch.Tensor([1/c if c != 0 else 0 for c in counts]).to(self.h_params.device)
            criterion_params['weight'] = freqs/torch.sum(freqs)

        if isinstance(self.supervised_lv, Categorical):
            self.criterion = nn.CrossEntropyLoss(**criterion_params)
        else:
            raise NotImplementedError('The supervision criterium has not been implemented yet '
                                      'for {} latent variables'.format(self.supervised_lv.name))

    def get_loss(self):
        num_classes = self.supervised_lv.size
        predictions = self.supervised_lv.post_params['logits'].view(-1, num_classes)
        target = self.net.variables_star[self.supervised_lv].view(-1)
        loss = self.criterion(predictions, target)

        self._prepare_metrics(loss)

        return loss

    def _prepare_metrics(self, loss):
        ce = loss
        with torch.no_grad():
            num_classes = self.supervised_lv.size
            predictions = self.supervised_lv.post_params['logits'].view(-1, num_classes)
            target = self.net.variables_star[self.supervised_lv].view(-1)
            prediction_mask = (target != self.supervised_lv.ignore).float()
            accuracy = torch.sum((torch.argmax(predictions, dim=-1) == target).float()*prediction_mask)
            accuracy /= torch.sum(prediction_mask)
        self._prepared_metrics = {'/{}_CE'.format(self.supervised_lv.name): ce,
                                  '/{}_accuracy'.format(self.supervised_lv.name): accuracy}


class ELBo(BaseCriterion):
    # This is actually Sticking The Landing (STL) ELBo, and not the standard one (it estimates the same quantity anyway)
    def __init__(self, model, w):
        super(ELBo, self).__init__(model, w)
        self.infer_net = model.infer_bn
        self.gen_net = model.gen_bn

        # Taking the variable that has no children as the target for generation
        self.generated_v = model.generated_v
        self.infer_lvs = {lv.name: lv for lv in self.infer_net.variables if lv.name != self.generated_v.name}
        self.gen_lvs = {lv.name: lv for lv in self.gen_net.variables if lv.name != self.generated_v.name}

        # Warning: This is still only implemented for categorical generation
        criterion_params = {'ignore_index': self.generated_v.ignore, 'reduction': 'none'}
        if self.generated_v.name in self.h_params.is_weighted:
            counts = [model.index[self.generated_v].freqs[w] for w in self.model.index[self.generated_v].itos]
            freqs = torch.Tensor([1/c if c != 0 else 0 for c in counts]).to(self.h_params.device)
            criterion_params['weight'] = freqs/torch.sum(freqs)
        self.criterion = nn.CrossEntropyLoss(**criterion_params)
        if self.generated_v.name in self.h_params.is_weighted:
            criterion_params.pop('weight')
        self._unweighted_criterion = nn.CrossEntropyLoss(**criterion_params)

        self.log_p_xIz = None
        self.log_p_z = None
        self.log_q_zIx = None

        self.sequence_mask = None
        self.valid_n_samples = None

    def get_loss(self, unweighted=False):
        # This probability is shaped like [batch_size]
        vocab_size = self.generated_v.size
        criterion = self._unweighted_criterion if unweighted else self.criterion
        self.sequence_mask = (self.gen_net.variables_star[self.generated_v] != self.generated_v.ignore).float()

        self.log_p_xIz = - criterion(self.gen_net.variables_hat[self.generated_v].view(-1, vocab_size),
                                     self.gen_net.variables_star[self.generated_v].view(-1)
                                     ).view(self.gen_net.variables_star[self.generated_v].shape) * self.sequence_mask

        self.valid_n_samples = torch.sum(self.sequence_mask)

        self.log_p_z = sum([self.gen_net.log_proba[lv] for lv in self.gen_lvs.values()]) * self.sequence_mask
        self.log_q_zIx = sum([self.infer_net.log_proba[lv] for lv in self.infer_lvs.values()]) * self.sequence_mask

        loss = - torch.sum(self.log_p_xIz + self.log_p_z - self.log_q_zIx, dim=(0, 1))/self.valid_n_samples
        if unweighted:
            unweighted_loss = loss
        else:
            un_log_p_xIz = self._unweighted_criterion(self.gen_net.variables_hat[self.generated_v].view(-1, vocab_size),
                                                      self.gen_net.variables_star[self.generated_v].view(-1)
                                       ).view(self.gen_net.variables_star[self.generated_v].shape) * self.sequence_mask
            unweighted_loss = - torch.sum(un_log_p_xIz + self.log_p_z - self.log_q_zIx, dim=(0, 1))/self.valid_n_samples
        self._prepare_metrics(unweighted_loss)

        return loss

    def _prepare_metrics(self, loss):
        current_elbo = - loss
        LL_name = '/p({}|{}'.format(self.generated_v.name, ', '.join([lv for lv in self.infer_lvs]))
        LL_value = torch.sum(self.log_p_xIz)/self.valid_n_samples
        KL_dict = {}
        for lv in self.gen_lvs.keys():
            gen_lv, inf_lv = self.gen_lvs[lv], self.infer_lvs[lv]
            infer_v_name = inf_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.infer_net.parent[inf_lv]]))
                                          if inf_lv in self.infer_net.parent else '')
            gen_v_name = gen_lv.name + ('I{}'.format(', '.join([lv.name for lv in self.gen_net.parent[gen_lv]]))
                                        if gen_lv in self.gen_net.parent else '')
            KL_name = '/KL(q({})IIp({}))'.format(infer_v_name, gen_v_name)
            log_q_lv = self.infer_net.log_proba[inf_lv] * self.sequence_mask
            log_p_lv = self.gen_net.log_proba[gen_lv] * self.sequence_mask
            KL_value = torch.sum(log_q_lv - log_p_lv)/self.valid_n_samples
            KL_dict[KL_name] = KL_value

        self._prepared_metrics = {'/ELBo': current_elbo, LL_name: LL_value, **KL_dict}


class IWLBo(ELBo):
    # This is actually DReG IWLBo and not the standard one (it estimates the same quantity anyway)
    def __init__(self, model, w):
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

    def _prepare_metrics(self):
        pass
