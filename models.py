import os
import abc


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from components import *
from h_params import *

# =================================================== EMBEDDING UTILITY ================================================


# ==================================================== BASE MODEL CLASS ================================================

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
        self.losses = nn.ModuleList([loss(self, *w) for loss, w in zip(h_params.losses, h_params.loss_weights)])

        # TODO: initialize network parameters

        # The Optimizer
        self.optimizer = h_params.optimizer(self.parameters, **h_params.optimizer_kwargs)

        # Getting the Summary writer
        self.writer = SummaryWriter(h_params.viz_path)

        # Initializing truth variables
        self.X = None
        self.Y = None
        self.iw_X = None
        self.iw_Y = None

        # Initializing estimated variables
        self.Z = [None for _ in h_params.z_types]
        self.Z_log_probas = [None for _ in h_params.z_types]
        self.Z_samples = [None for _ in h_params.z_types]
        self.iw_Z = [None for _ in h_params.z_types]
        self.iw_Z_log_probas = [None for _ in h_params.z_types]
        self.iw_Z_samples = [None for _ in h_params.z_types]

        self.X_hat_params = None
        self.iw_X_hat_params = None

        self.iw_samples = None

    def opt_step(self, samples):

        if (self.step % self.h_params.grad_accumulation_steps) == 0:
            # Reinitializing gradients if accumulation is over
            self.optimizer.zero_grad()

        # Forward and backward pass
        self.forward(samples)
        if isinstance(self, ImportanceWeightedAEMixin):
            self.iw_forward(samples, is_training=True)

        losses = [loss.forward(samples) * loss.w for loss in self.losses]
        sum(losses).backward()

        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            # Applying gradients if accumulation is over
            self.optimizer.step()

        self._dump_train_viz()
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
            for name, metric in loss.metrics(is_training=True):
                self.writer.add_scalar(os.path.join('train', name), metric, self.step)

    def dump_test_viz(self, gen_samples=1):
        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for name, metric in loss.metrics(is_training=False):
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
# ==================================================== AE MIX-INS ======================================================

class DeterministicAEMixin(BaseAE, metaclass=abc.ABCMeta):
    def forward(self, samples):
        # Samples should contain [X, Y]
        self.X = samples[0]
        if len(samples) > 1:  # Checking whether we have labels
            self.Y = samples[1]

        z_params = []
        z_log_probas = []
        z_samples = []
        for x in self.X:
            z_params_i, _, _ = self.encoder.forward(x, z_samples[-1])
            posterior = [z.posterior_sample(params) for z, params in zip(z_params_i, self.h_params.z_types)]
            z_samples.append(torch.cat([pos_i[0] for pos_i in posterior]))
            z_log_probas.append(torch.cat([pos_i[1] for pos_i in posterior]))
            z_params.append(z_params_i)

        # Saving the current latent variable posterior samples in the class attributes and linking them to the
        # back-propagation graph
        # For DAE, the samples are the m parameter itself for gaussians, and softmax(logits, temperature) for
        # categorical variables
        self.Z_samples = nn.ModuleList([
            z_params_i['loc'] if 'loc' in z_params_i else F.softmax(z_params_i['logits']) if 'logits' in z_params_i
            else None
            for z_params_i in z_params])
        self.Z_log_probas = nn.ModuleList(z_log_probas)
        self.Z = nn.ModuleList(z_params)

        x_hat_params = []
        x_prev = self.word_embeddings[self.vocab_index.w2i['<go>']]
        for z_sample, x_i in zip(self.Z_samples, self.X):
            x_hat_params_i, _, _ = self.decoder.forward(z_sample, x_prev)
            x_hat_params.append(x_hat_params_i)
            x_prev = x_i

        self.X_hat_params = nn.ModuleList(x_hat_params)


class VariationalAEMixin(BaseAE, metaclass=abc.ABCMeta):
    def forward(self, samples):
        # Samples should contain [X, Y]
        self.X = samples[0]
        if len(samples) > 1:  # Checking whether we have labels
            self.Y = samples[1]

        z_params = []
        z_log_probas = []
        z_samples = []
        for x in self.X:
            z_params_i, _, _ = self.encoder.forward(x, z_samples[-1])
            posterior = [z.posterior_sample(params) for z, params in zip(z_params_i, self.h_params.z_types)]
            z_samples.append(torch.cat([pos_i[0] for pos_i in posterior]))
            z_log_probas.append(torch.cat([pos_i[1] for pos_i in posterior]))
            z_params.append(z_params_i)

        # Saving the current latent variable posterior samples in the class attributes and linking them to the
        # back-propagation graph
        self.Z_samples = nn.ModuleList(z_samples)
        self.Z_log_probas = nn.ModuleList(z_log_probas)
        self.Z = nn.ModuleList(z_params)

        x_hat_params = []
        x_prev = self.word_embeddings[self.vocab_index.w2i['<go>']]
        for z_sample, x_i in zip(self.Z_samples, self.X):
            x_hat_params_i, _, _ = self.decoder.forward(z_sample, x_prev)
            x_hat_params.append(x_hat_params_i)
            x_prev = x_i

        self.X_hat_params = nn.ModuleList(x_hat_params)


class ImportanceWeightedAEMixin(BaseAE, metaclass=abc.ABCMeta):
    def iw_forward(self, samples, is_training):
        if is_training:
            self.iw_samples = self.h_params.training_iw_samples
        else:
            self.iw_samples = self.h_params.testing_iw_samples
        # Samples should contain [X, Y]
        x_size = samples[0].shape
        self.iw_X = samples[0].expand(x_size[0]*self.iw_samples, *x_size[1:])
        if len(samples) > 1:  # Checking whether we have labels
            y_size = samples[1].shape
            self.iw_Y = samples[1].expand(y_size[0]*self.iw_samples, *y_size[1:])

        z_params = []
        z_log_probas = []
        z_samples = []
        for x in self.iw_X:
            z_params_i, _, _ = self.encoder.forward(x, z_samples[-1])
            posterior = [z.posterior_sample(params) for z, params in zip(z_params_i, self.h_params.z_types)]
            z_samples.append(torch.cat([pos_i[0] for pos_i in posterior]))
            z_log_probas.append(torch.cat([pos_i[1] for pos_i in posterior]))
            z_params.append(z_params_i)

        # Saving the current latent variable posterior samples in the class attributes and linking them to the
        # back-propagation graph
        self.iw_Z_samples = nn.ModuleList(z_samples)
        self.iw_Z_log_probas = nn.ModuleList(z_log_probas)
        self.iw_Z = nn.ModuleList(z_params)

        x_hat_params = []
        x_prev = self.word_embeddings[self.vocab_index.w2i['<go>']]
        for z_sample, x_i in zip(self.Z_samples, self.X):
            x_hat_params_i, _, _ = self.decoder.forward(z_sample, x_prev)
            x_hat_params.append(x_hat_params_i)
            x_prev = x_i

        self.iw_X_hat_params = nn.ModuleList(x_hat_params)


# ======================================================================================================================
# ==================================================== MODEL CLASSES ===================================================

class SSAE(BaseAE, DeterministicAEMixin):
    def __init__(self, vocab_index, h_params=None):
        if h_params is None:
            h_params = DefaultSSHParams
        super(SSAE, self).__init__(vocab_index, h_params)
        assert any([isinstance(loss, Supervision) for loss in self.losses]), "You forgot to include a supervision loss."


class AE(BaseAE, DeterministicAEMixin):
    def __init__(self, vocab_index, h_params=None):
        if h_params is None:
            h_params = DefaultHParams
        super(AE, self).__init__(vocab_index, h_params)


class VAE(BaseAE, VariationalAEMixin):
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

# ======================================================================================================================
