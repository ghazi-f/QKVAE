import math
import abc
from collections import defaultdict

import torch
import torch.nn as nn

from components.latent_variables import BaseLatentVariable, Categorical, Gaussian
from components.links import BaseLink

from time import time

class BayesNet(nn.Module):
    def __init__(self, vertices):
        super(BayesNet, self).__init__()
        for p, l, c in vertices:
            assert isinstance(p, BaseLatentVariable) and isinstance(l, BaseLink) and isinstance(c, BaseLatentVariable)
        # TODO: Also check that the vertices don't have cycles

        self.vertices = vertices
        self.parent = defaultdict(list)
        self.child = defaultdict(list)
        self.approximator = {}
        self.name_to_v = {}
        # Note: If a child has multiple parents, the same link must be provided between all the parents and the
        # same child
        for p, l, c in vertices:
            self.parent[c].append(p)
            self.child[p].append(c)
            self.approximator[c] = l
            self.name_to_v[p.name] = p
            self.name_to_v[c.name] = c
        self.input_variables = []
        for variables in self.parent.values():
            for variable in variables:
                if variable not in self.parent and variable not in self.input_variables:
                    self.input_variables.append(variable)

        self.variables_hat = {}
        self.variables_star = {}
        self.variables = set(list(self.parent.keys()) + list(self.child.keys()))
        self.log_proba = {lv: None for lv in self.variables}

        self.iw = any([lv.iw for lv in self.variables])
        if self.iw:
            # Building duplication levels with regard to importance weighted variables
            self.dp_lvl = {}
            for lv in self.variables:
                lvl = 0
                if lv in self.parent:
                    lvl = max([self._get_max_iw_path(p, lvl) for p in self.parent[lv]])
                self.dp_lvl[lv] = lvl
        else:
            self.dp_lvl = {lv: 0 for lv in self.variables}

    def _get_max_iw_path(self, lv, lvl, force_lv=None):
        if (lv.iw and force_lv is None) or (force_lv and lv.name in force_lv):
            lvl += 1
        if lv not in self.parent:
            return lvl
        else:
            return max([self._get_max_iw_path(p, lvl, force_lv=force_lv) for p in self.parent[lv]])

    def clear_values(self):
        self.variables_hat = {}
        self.variables_star = {}
        self.log_proba = {lv: None for lv in self.variables}
        for var in self.variables:
            var.clear_values()

    def forward(self, inputs, n_iw=None, target=None, eval=False, prev_states=None, force_iw=None, complete=False,
                lens=None):
        # The forward pass propagates the root variable values yielding
        if prev_states is None:
            prev_states = {v: None for v in self.variables}

        # Getting current duplication levels
        if force_iw:
            dp_lvl = {}
            for lv in self.variables:
                lvl = 0
                if lv in self.parent:
                    lvl = max([self._get_max_iw_path(p, lvl, force_iw) for p in self.parent[lv]])
                dp_lvl[lv] = lvl
        else:
            dp_lvl = self.dp_lvl
        # Loading the inputs into the network
        self.clear_values()
        for lv in self.variables:
            if lv.name in inputs:
                self.variables_star[lv] = inputs[lv.name]

        # Checking that all the inputs have been given
        for lv in self.input_variables:
            assert lv in self.variables_star or lv.allow_prior, "You didn't provide a value for {} ".format(lv.name)
            if lv not in self.variables_star:
                self.variables_hat[lv], self.log_proba[lv] = lv.prior_sample(list(inputs.values())[0].shape[:-1])
            elif lv.allow_prior:
                self.variables_hat[lv], _ = lv.prior_sample(list(inputs.values())[0].shape[:-1])
                self.log_proba[lv] = lv.prior_log_prob(self.variables_star[lv])
            lv.post_reps, lv.post_samples = lv.post_reps or lv.prior_reps,  lv.post_samples or lv.prior_samples
            lv.post_log_probas, lv.post_params = lv.post_log_probas or lv.prior_log_probas, lv.post_params or lv.prior_params
        if target is not None:
            # Collecting requirements to estimate the target
            lvs_to_fill = [target]
            collected = False
            while not collected:
                collected = True
                for lv in lvs_to_fill:
                    for p in self.parent[lv]:
                        if (p not in lvs_to_fill) and (p not in self.variables_star):
                            collected = False
                            lvs_to_fill.append(p)
        else:
            lvs_to_fill = self.parent.keys()

        # Ancestral sampling from the network
        while any([lv not in self.variables_hat for lv in lvs_to_fill]):
            # Going through all the unfilled child variables
            for lv in lvs_to_fill:
                parents_available = all([(p in self.variables_star) or (p in self.variables_hat)
                                         for p in self.parent[lv]])
                still_unfilled = lv not in self.variables_hat
                if parents_available and still_unfilled:
                    # Gathering conditioning variables
                    max_cond_lvl = dp_lvl[lv]
                    lv_conditions = {p.name: self._ready_condition(p, n_iw, max_cond_lvl, prev_states, dp_lvl, force_iw,
                                                                   eval)
                                     for p in self.parent[lv]}

                    # Setting up ground truth to be injected if any
                    gt_lv = self.variables_star[lv] if lv in self.variables_star else None

                    # Repeating inputs if the latent variable is importance weighted
                    if ((lv.iw and force_iw is None) or (force_iw and lv.name in force_iw)) and n_iw is not None:
                        for k, v in lv_conditions.items():
                            expand_arg = [n_iw]+list(v.shape)
                            lv_conditions[k] = v.unsqueeze(0).expand(expand_arg)
                        if gt_lv is not None:
                            expand_arg = [n_iw]+list(gt_lv.shape)
                            gt_lv = gt_lv.unsqueeze(0).expand(expand_arg)
                            for _ in range(max_cond_lvl):
                                expand_arg = [n_iw]+list(gt_lv.shape)
                                gt_lv = gt_lv.unsqueeze(0).expand(expand_arg)
                        if lens is not None:
                            expand_arg = [n_iw] + list(lens.shape)
                            this_len = lens.unsqueeze(0).expand(expand_arg)
                            for _ in range(max_cond_lvl):
                                expand_arg = [n_iw] + list(this_len.shape)
                                this_len = this_len.unsqueeze(0).expand(expand_arg)
                            this_len = this_len.reshape(-1)
                        else:
                            this_len = lens
                    else:
                        this_len = lens
                    lv(self.approximator[lv], lv_conditions, gt_samples=gt_lv, complete=(lv in self.child) or complete,
                       lens=this_len)
                    if eval:
                        if isinstance(lv, Categorical):
                            self.variables_hat[lv] = torch.nn.functional.one_hot(torch.argmax(lv.post_params['logits'],
                                                                                              dim=-1), lv.size)
                        elif isinstance(lv, Gaussian):
                            self.variables_hat[lv] = lv.post_params['loc']
                        else:
                            raise NotImplementedError('Unidentifiable latent variable type {} for variable '
                                                      '{}'.format(type(lv), lv.name))
                    else:
                        self.variables_hat[lv] = lv.post_samples
                    self.log_proba[lv] = lv.post_gt_log_probas if gt_lv is not None else lv.post_log_probas

        if target is None:
            assert all([lv in self.variables_hat or lv in self.variables_star for lv in self.variables])
            assert all([lv in self.log_proba or (lv in self.input_variables and not lv.allow_prior)
                        for lv in self.variables])
        new_prev_state = {v: tuple(v_i.detach() for v_i in v.prev_state) if v.prev_state is not None else None
                          for v in self.variables}
        return new_prev_state

    def _ready_condition(self, lv, n_iw, max_lvl, prev_states, dp_lvl, force_iw, eval):
        value = lv.rep(self.variables_star[lv], step_wise=False, prev_rep=prev_states[lv])\
                if lv in self.variables_star \
                else lv.rep(self.variables_hat[lv], step_wise=False, prev_rep=prev_states[lv]) if eval \
                else lv.post_reps
        if n_iw is not None and n_iw > 1:
            for _ in range(dp_lvl[lv] + (1 if (lv.iw or lv.name in (force_iw or [])) else 0), max_lvl):
                expand_arg = [n_iw] + list(value.shape)
                value = value.unsqueeze(0).expand(expand_arg)
        return value

    def prior_sample(self, sample_shape):
        self.clear_values()

        # Getting all the latent variables that have no parents (roots), and that, consequently, need to be sampled from
        # their respective priors.
        roots = [lv for lv in self.child.keys() if lv not in self.parent]
        inputs = {}
        for lv in roots:
            inputs[lv.name] = lv.prior_sample(sample_shape)[0]

        self(inputs)
