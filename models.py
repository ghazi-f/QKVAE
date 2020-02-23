import os
import abc


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from components import *
from h_params import *

# =================================================== EMBEDDING UTILITY ================================================


# =========================================== BASE SEMI SUPERVISED MODEL CLASS =========================================
class BaseAE(nn.module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, h_params):
        super(BaseAE, self).__init__()

        self.h_params = h_params
        self.vocab_index = vocab_index
        self.word_embeddings = nn.Embedding(h_params.vocab_size, h_params.embedding_dim)

        # Encoder
        self.encoder = h_params.encoder(h_params, self.word_embeddings)

        # Decoder components
        self.decoder = h_params.decoder(h_params, self.word_embeddings)

        # The losses
        self.losses = nn.ModuleList([loss(self, w) for loss, w in zip(h_params.losses, h_params.loss_weights)])

        # TODO: initialize network parameters

        # The Optimizer
        self.optimizer = h_params.optimizer(self.parameters, **h_params.optimizer_kwargs)

        # Getting the Summary writer
        self.writer = SummaryWriter(h_params.viz_path)

    def opt_step(self, samples):

        if (self.step % self.h_params.grad_accumulation_steps) == 0:
            self.optimizer.zero_grad()
        sum([loss.forward(samples) * loss.w for loss in self.losses]).backward()
        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            self.optimizer.step()
        self._dump_train_visu()
        self.step += 1

    def prior_sample(self, sample_size):
        # unrolling the decoder layer
        # TODO: move this to the VAE daughter of this class
        pass

    def _dump_train_viz(self):
        # Dumping gradient norm
        grad_norm = 0
        for module, name in zip([self, self.encoder, self.decoder], ['overall', 'encoder', 'decoder']):
            for p in module.parameters():
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1. / 2)
            self.writer.add_scalar(os.path.join('train', '_'.join([name, 'grad_norm'])), grad_norm, self.step)

        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for metric, name in loss.metrics():
                self.writer.add_scalar(os.path.join('train', name), metric, self.step)

    def dump_test_viz(self, samples, gen_samples=1):
        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for metric, name in loss.metrics(samples):
                self.writer.add_scalar(os.path.join('test', name), metric, self.step)

    def decode_to_text(self, z):
        # It is assumed that this function is used at test time for display purposes
        outputs, lens = self.decoder.forward(z, is_training=False)
        text = [self.vocab_index.i2w[output[:l]] for output, l in zip(outputs, lens)]
        return text

    def save(self):
        torch.save(self.state_dict(), self.h_params.save_path)
        print("Model {} saved !".format(self.h_params.test_name))

    def load(self):
        self.load_state_dict(torch.load(self.h_params.save_path))


# ======================================================================================================================
class SSAE(BaseAE):
    def __init__(self, vocab_index, h_params=None):
        if h_params is None:
            h_params = DefaultSSHParams
        super(SSAE, self).__init__(vocab_index, h_params)
        assert any([isinstance(loss, Supervision) for loss in self.losses]), "You forgot to include a supervision loss."


class AE(BaseAE):
    def __init__(self, vocab_index, h_params=None):
        if h_params is None:
            h_params = DefaultHParams
        super(AE, self).__init__(vocab_index, h_params)


class VAE(BaseAE):
    def __init__(self, vocab_index, h_params=None):
        if h_params is None:
            h_params = DefaultVariationalHParams
        super(VAE, self).__init__(vocab_index, h_params)
        assert any([isinstance(loss, ELBo) for loss in self.losses]), "There appears to be no ELBo in your losses"
        assert not any([not isinstance(loss, ELBo) and isinstance(loss, Reconstruction)
                        for loss in self.losses]), "A non variational sample reconstruction loss is in your losses"

class SSVAE(VAE, SSAE):
    def __init__(self, vocab_index, h_params=None):
        if h_params is None:
            h_params = DefaultSSVariationalHParams
        super(VAE, self).__init__(vocab_index, h_params)
