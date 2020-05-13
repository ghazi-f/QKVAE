import math
import abc
from collections import defaultdict

import torch
import torch.nn as nn

from components.latent_variables import BaseLatentVariable, Categorical
from components.links import BaseLink


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

    def _get_max_iw_path(self, lv, lvl):
        if lv.iw:
            lvl += 1
        if lv not in self.parent:
            return lvl
        else:
            return max([self._get_max_iw_path(p, lvl) for p in self.parent[lv]])

    def clear_values(self):
        self.variables_hat = {}
        self.variables_star = {}
        self.log_proba = {lv: None for lv in self.variables}

    def forward(self, inputs, n_iw=None, target=None):
        # The forward pass propagates the root variable values yielding
        # assert n_iw is not None or not self.iw, "You didn't provide a number of importance weights."

        # Loading the inputs into the network
        self.clear_values()
        for lv in self.variables:
            if lv.name in inputs:
                self.variables_star[lv] = inputs[lv.name]

        # Checking that all the inputs have been given
        for lv in self.input_variables:
            assert lv in self.variables_star, "You didn't provide a value for {} ".format(lv.name)
            if lv.allow_prior:
                self.log_proba[lv] = lv.prior_log_prob(self.variables_star[lv])
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
                    max_cond_lvl = max([self.dp_lvl[p] for p in self.parent[lv]])
                    lv_conditions = torch.cat([self._ready_condition(p, n_iw, max_cond_lvl) for p in self.parent[lv]],
                                              dim=-1)

                    # Setting up ground truth to be injected if any
                    gt_lv = self.variables_star[lv] if lv in self.variables_star else None

                    # Repeating inputs if the latent variable is importance weighted
                    if lv.iw and n_iw is not None:
                        expand_arg = [n_iw]+list(lv_conditions.shape)
                        lv_conditions = lv_conditions.unsqueeze(0).expand(expand_arg)
                        if gt_lv is not None:
                            expand_arg = [n_iw]+list(gt_lv.shape)
                            gt_lv = gt_lv.unsqueeze(0).expand(expand_arg)
                            for _ in range(max_cond_lvl):
                                expand_arg = [n_iw]+list(gt_lv.shape)
                                gt_lv = gt_lv.unsqueeze(0).expand(expand_arg)
                    lv(self.approximator[lv], lv_conditions, gt_samples=gt_lv)
                    self.variables_hat[lv] = lv.post_samples
                    self.log_proba[lv] = lv.post_gt_log_probas if gt_lv is not None else lv.post_log_probas

        if target is None:
            assert all([lv in self.variables_hat or lv in self.variables_star for lv in self.variables])
            assert all([lv in self.log_proba or (lv in self.input_variables and not lv.allow_prior)
                        for lv in self.variables])

    def _ready_condition(self, lv, n_iw, max_lvl):
        value = lv.rep(self.variables_star[lv], step_wise=False) if lv in self.variables_star else lv.post_reps
        if n_iw is not None and n_iw > 1:
            for _ in range(self.dp_lvl[lv], max_lvl):
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
