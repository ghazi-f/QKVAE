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

class BaseAE(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, h_params, autoload=True):
        super(BaseAE, self).__init__()

        self.h_params = h_params
        self.vocab_index = vocab_index
        self.word_embeddings = nn.Embedding(h_params.vocab_size, h_params.embedding_dim)

        # Encoder
        self.encoder = h_params.encoder(h_params, self.word_embeddings)

        # Linking latent variables to the main Model (mainly to propagate the 'device' property)
        self.z_types = nn.ModuleList(h_params.z_types)

        # Decoder components
        self.decoder = h_params.decoder(h_params, self.word_embeddings)

        # The losses
        self.losses = [loss(self, *w) for loss, w in zip(h_params.losses, h_params.loss_params)]

        # TODO: initialize network parameters

        # The Optimizer
        self.optimizer = h_params.optimizer(self.parameters(), **h_params.optimizer_kwargs)

        # Getting the Summary writer
        self.writer = SummaryWriter(h_params.viz_path)
        self.step = 0

        # Defining placeholders for ground truth variables
        self.X = None
        self.X_lens = None
        self.Y = None
        self.iw_X = None
        self.iw_Y = None

        # Defining placeholders for estimated variables
        self.Z_params = [[None for _ in h_params.z_types] for _ in range(self.h_params.batch_size)]
        self.Z_log_probas = [None for _ in h_params.z_types]
        self.Z_samples = [None for _ in h_params.z_types]
        self.iw_Z_params = [None for _ in h_params.z_types]
        self.iw_Z_log_probas = [None for _ in h_params.z_types]
        self.iw_Z_samples = [None for _ in h_params.z_types]

        self.X_hat_params = None
        self.iw_X_hat_params = None

        self.iw_samples = None

        # Loading previous checkpoint if auto_load is set to True
        if autoload:
            self.load()

    def opt_step(self, samples):

        if (self.step % self.h_params.grad_accumulation_steps) == 0:
            # Reinitializing gradients if accumulation is over
            self.optimizer.zero_grad()

        # Forward and backward pass
        self.forward(samples, is_training=True)
        if isinstance(self, ImportanceWeightedAEMixin):
            self.iw_forward(samples, is_training=True)

        losses = [loss.get_loss() * loss.w for loss in self.losses]
        sum(losses).backward()

        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            # Applying gradients if accumulation is over
            self.optimizer.step()
            self.step += 1

        self._dump_train_viz()

        return sum(losses)

    def _dump_train_viz(self):
        # Dumping gradient norm
        grad_norm = 0
        for module, name in zip([self, self.encoder, self.decoder], ['overall', 'encoder', 'decoder']):
            for p in module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1. / 2)
            self.writer.add_scalar('train' + '/' + '_'.join([name, 'grad_norm']), grad_norm, self.step)

        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for name, metric in loss.metrics(is_training=True):
                self.writer.add_scalar('train' + name, metric, self.step)

    def dump_test_viz(self, complete=False):
        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for name, metric in loss.metrics(is_training=True):
                self.writer.add_scalar('test' + name, metric, self.step)

        summary_dumpers = {'scalar':self.writer.add_scalar, 'text': self.writer.add_text,
                           'image': self.writer.add_image}

        # We limit the generation of these samples to the less frequent "complete" test visualisations because their
        # computational cost may be high, and because the make the log file a lot larger.
        if complete:
            for summary_type, summary_name, summary_data in self.data_specific_metrics():
                summary_dumpers[summary_type]('test'+summary_name, summary_data  , self.step)

    def data_specific_metrics(self):
        # this is supposed to output a list of (summary type, summary name, summary data) triplets
        return []

    def save(self):
        root = ''
        for subfolder in self.h_params.save_path.split(os.sep)[:-1]:
            root = os.path.join(root, subfolder)
            if not os.path.exists(root):
                os.mkdir(root)
        torch.save({'model_checkpoint': self.state_dict(), 'step': self.step}, self.h_params.save_path)
        print("Model {} saved !".format(self.h_params.test_name))

    def load(self):
        if os.path.exists(self.h_params.save_path):
            checkpoint = torch.load(self.h_params.save_path)
            model_checkpoint, self.step = checkpoint['model_checkpoint'], checkpoint['step']
            self.load_state_dict(model_checkpoint)
            print("Loaded model at step", self.step)
        else:
            print("Save file doesn't exist, the model will be trained from scratch.")

# ======================================================================================================================
# ==================================================== AE MIX-INS ======================================================

class DeterministicAEMixin:
    def forward(self, samples, is_training):
        # Samples should contain [X, Y] if supervised, else [X] if unsupervised
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
        self.Z_params = nn.ModuleList(z_params)

        x_hat_params = []
        x_prev = self.word_embeddings[self.vocab_index.stoi['<go>']]
        for z_sample, x_i in zip(self.Z_samples, self.X):
            x_hat_params_i, _, _ = self.decoder.forward(z_sample, x_prev)
            x_hat_params.append(x_hat_params_i)
            x_prev = x_i

        self.X_hat_params = nn.ModuleList(x_hat_params)


class VariationalAEMixin:
    def forward(self, samples, is_training):
        # Samples should contain [X, Y]
        self.X = samples[0][0]
        self.X_lens = samples[0][1]
        if len(samples) > 1:  # Checking whether we have labels
            self.Y = samples[1]

        z_params = []
        z_log_probas = []
        z_samples = []
        for x in self.X.transpose(0, 1):
            # Inputs are unsqueezed to have single element time-step dimension, while previous outputs are unsqueezed to
            # have a single element num_layers*num_directions dimension (to be changed for multilayer GRUs)
            z_params_i, _, _ = self.encoder.forward(x.unsqueeze(1), z_samples[-1] if len(z_samples) else None)
            posterior = [z.posterior_sample(params) for params, z in zip(z_params_i, self.h_params.z_types)]
            z_samples.append(torch.cat([pos_i[0] for pos_i in posterior], dim=-1))
            z_log_probas.append(torch.stack([pos_i[1] for pos_i in posterior]))
            z_params.append(z_params_i)

        # Saving the current latent variable posterior samples in the class attributes and linking them to the
        # back-propagation graph
        self.Z_samples = torch.stack(z_samples).transpose(0, 1)
        self.Z_log_probas = torch.stack(z_log_probas).transpose(0, 2)
        self.Z_params = z_params  # Careful about this one, it's shape is (time-step, z-types, params, batch, features)

        x_hat_params = []
        go_tokens = (torch.ones(self.X.size(0)).long()*self.vocab_index.stoi['<go>']).to(self.h_params.device)
        x_prev = self.word_embeddings(go_tokens)
        for z_sample, x_i in zip(self.Z_samples.transpose(0, 1), self.X.transpose(0, 1)):
            # Inputs are unsqueezed to have single element time-step dimension, while previous outputs are unsqueezed to
            # have a single element num_layers*num_directions dimension (to be changed for multilayer GRUs)
            x_hat_params_i, _, _ = self.decoder.forward(z_sample.unsqueeze(1), x_prev)
            x_hat_params.append(x_hat_params_i)
            if is_training:
                x_prev =self.word_embeddings(x_i)
            else:
                self.word_embeddings(torch.argmax(x_hat_params[-1], dim=-1))

        self.X_hat_params = torch.stack(x_hat_params).transpose(0, 1)

    def prior_sample(self, n_samples):

        z_samples = torch.cat([z_t_i.prior_sample((n_samples, self.h_params.max_len))[0] for z_t_i in self.h_params.z_types],
                              dim=-1)

        x_hat_params = []
        go_tokens = (torch.ones(n_samples).long()*self.vocab_index.stoi['<go>']).to(self.h_params.device)
        x_prev = self.word_embeddings(go_tokens)
        for z_sample, x_i in zip(z_samples.transpose(0, 1), self.X.transpose(0, 1)):
            # Inputs are unsqueezed to have single element time-step dimension, while previous outputs are unsqueezed to
            # have a single element num_layers*num_directions dimension (to be changed for multilayer GRUs)
            x_hat_params_i, _, _ = self.decoder.forward(z_sample.unsqueeze(1), x_prev)
            x_hat_params.append(x_hat_params_i)
            self.word_embeddings(torch.argmax(x_hat_params[-1], dim=-1))

        sentences = torch.stack(x_hat_params).transpose(0, 1)

        return sentences

    def decode_to_text(self, x_hat_params):
        # It is assumed that this function is used at test time for display purposes
        # Getting the argmax from the one hot if it's not done
        if x_hat_params.shape[-1] == self.h_params.vocab_size:
            x_hat_params = torch.argmax(x_hat_params, dim=-1)
        text = '||'.join([' '.join([self.vocab_index.itos[x_i_h_p_j] for x_i_h_p_j in x_i_h_p])
                          for x_i_h_p in x_hat_params])
        return text

    def data_specific_metrics(self):
        # this is supposed to output a list of (summary type, summary name, summary data) triplets
        with torch.no_grad():
            summary_triplets = [
                ('text', '/ground_truth', self.decode_to_text(self.X)),
                ('text', '/reconstructions', self.decode_to_text(self.X_hat_params)),
                ('text', '/prior_sample', self.decode_to_text(self.prior_sample(self.h_params.test_prior_samples)))
            ]

        return summary_triplets


class ImportanceWeightedAEMixin:
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
        x_prev = self.word_embeddings[self.vocab_index.stoi['<go>']]
        for z_sample, x_i in zip(self.Z_samples, self.X):
            x_hat_params_i, _, _ = self.decoder.forward(z_sample, x_prev)
            x_hat_params.append(x_hat_params_i)
            x_prev = x_i

        self.iw_X_hat_params = nn.ModuleList(x_hat_params)


# ======================================================================================================================
# ==================================================== MODEL CLASSES ===================================================

class SSAE(BaseAE, DeterministicAEMixin):
    def __init__(self, vocab_index, h_params=None):
        super(SSAE, self).__init__(vocab_index, h_params)
        assert any([isinstance(loss, Supervision) for loss in self.losses]), "You forgot to include a supervision loss."


class AE(BaseAE, DeterministicAEMixin):
    def __init__(self, vocab_index, h_params=None):
        super(AE, self).__init__(vocab_index, h_params)


class VAE(BaseAE, VariationalAEMixin):
    def __init__(self, vocab_index, h_params=None):
        super(VAE, self).__init__(vocab_index, h_params)
        assert any([isinstance(loss, ELBo) for loss in self.losses]), "There appears to be no ELBo in your losses"
        assert not any([not isinstance(loss, ELBo) and isinstance(loss, Reconstruction)
                        for loss in self.losses]), "A non variational sample reconstruction loss is in your losses"


class SSVAE(VariationalAEMixin, BaseAE):
    def __init__(self, vocab_index, h_params=None):
        super(SSVAE, self).__init__(vocab_index, h_params)

# ======================================================================================================================
