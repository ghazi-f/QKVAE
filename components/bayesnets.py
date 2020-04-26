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

    def clear_values(self):
        self.variables_hat = {}
        self.variables_star = {}
        self.log_proba = {lv: None for lv in self.variables}

    def forward(self, inputs):
        # The forward pass propagates the root variable values yielding

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

        # Ancestral sampling from the network
        while any([lv not in self.variables_hat for lv in self.parent.keys()]):
            # Going through all the unfilled child variables
            for lv in self.parent.keys():
                parents_available = all([p in self.variables_star or p in self.variables_hat for p in self.parent[lv]])
                still_unfilled = lv not in self.variables_hat
                if parents_available and still_unfilled:
                    lv_conditions = torch.cat([p.rep(self.variables_star[p], step_wise=False)
                                               if p in self.variables_star else p.post_reps for p in self.parent[lv]],
                                              dim=-1)

                    gt_lv = self.variables_star[lv] if lv in self.variables_star else None
                    lv(self.approximator[lv], lv_conditions, gt_samples=gt_lv)
                    self.variables_hat[lv] = lv.post_samples
                    self.log_proba[lv] = lv.post_gt_log_probas if gt_lv is not None else lv.post_log_probas

        assert all([lv in self.variables_hat or lv in self.variables_star for lv in self.variables])
        assert all([lv in self.log_proba or (lv in self.input_variables and not lv.allow_prior)
                    for lv in self.variables])

    def prior_sample(self, sample_shape):
        self.clear_values()

        # Getting all the latent variables that have no parents (roots), and that, consequently, need to be sampled from
        # their respective priors.
        roots = [lv for lv in self.child.keys() if lv not in self.parent]
        inputs = {}
        for lv in roots:
            inputs[lv.name] = lv.prior_sample(sample_shape)[0]

        self.forward(inputs)
