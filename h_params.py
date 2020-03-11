# Design this file with a hierarchical experiment checkpointing scheme
import os

from torch import optim
from torch import nn
from components import *

ROOT_CHECKPOINTING_PATH = 'checkpoints'
ROOT_TENSORBOARD_PATH = 'tb_logs'


# ======================================================================================================================
# ==================================================== HYPER-PARAMETERS ================================================
class DefaultHParams:
    def __init__(self, vocab_size,
                 max_len,
                 batch_size,
                 n_epochs,
                 device=None,
                 test_name='default',
                 embedding_dim=100,
                 encoder=GRUEncoder,
                 encoder_h=128,
                 encoder_l=2,
                 z_types=None,
                 decoder_h=32,
                 decoder_l=2,
                 decoder=GRUDecoder,
                 x_var=None,
                 losses=None,
                 loss_params=None,
                 optimizer=optim.Adam,
                 optimizer_kwargs=None,
                 grad_accumulation_steps=1,
                 training_iw_samples=10,
                 testing_iw_samples=100,
                 test_prior_samples=5,
                 weight_reconstruction=False
                 ):
        # A name to be used for checkpoints and Tensorboard logging indexation
        self.test_name = test_name
        self.save_path = os.path.join(ROOT_CHECKPOINTING_PATH, test_name+'.pth')
        self.viz_path = os.path.join(ROOT_TENSORBOARD_PATH, test_name)

        # Device hyper-parameter
        self.device = device or torch.device('cpu')

        # Data related hyper-parameters
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Corpus related hyper-parameters
        self.vocab_size = vocab_size

        # Architectural hyper-parameters
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        self.encoder = encoder
        self.encoder_h = encoder_h
        self.encoder_l = encoder_l

        # IDEA: Maybe define z variables through a dictionary to describe tree structured latent variables
        self.z_types = z_types or [Categorical(10, 'y_hat', self.device), Gaussian(100, 'z', self.device)]

        self.decoder_h = decoder_h
        self.decoder_l = decoder_l
        self.decoder = decoder

        self.x_var = x_var or Categorical(vocab_size, "X_hat", device)

        # Specifying losses
        self.losses = losses or [Reconstruction]
        self.loss_params = loss_params or [
            [1]]  # [weight] for unsupervised losses, and [weight, supervised_z_index] for the supervised losses
        self.weight_reconstruction = weight_reconstruction

        # Optimization hyper-parameters
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3}
        self.grad_accumulation_steps = grad_accumulation_steps

        # Sampling hyper_parameters
        self.training_iw_samples = training_iw_samples
        self.testing_iw_samples = testing_iw_samples
        self.test_prior_samples = test_prior_samples


        # This constructing will mainly serve as a sanity-check for the hyper parameter setting
        assert len(self.losses) == len(self.loss_params)
        assert 'lr' in self.optimizer_kwargs


class DefaultVariationalHParams(DefaultHParams):
    def __init__(self, vocab_size, max_len, batch_size, n_epoch, device=None):
        super(DefaultVariationalHParams, self).__init__(vocab_size, max_len, batch_size, n_epoch,
                                                        test_name='defaultV',
                                                        device=device,
                                                        losses=[ELBo],
                                                        loss_params=[[1]])


class DefaultSSHParams(DefaultHParams):
    def __init__(self, vocab_size, max_len, batch_size, n_epoch, device=None):
        super(DefaultSSHParams, self).__init__(vocab_size, max_len, batch_size, n_epoch,
                                               test_name='defaultSS',
                                               device=device,
                                               losses=[Supervision, Reconstruction],
                                               loss_params=[[1, 0], [1]])


class DefaultSSVariationalHParams(DefaultHParams):
    def __init__(self, vocab_size, max_len, batch_size, n_epochs, supervision_size, device=None,
                 target_ignore_index=None, token_ignore_index=None, **kwargs):
        default_kwargs = {'vocab_size': vocab_size,
                          'max_len': max_len,
                          'batch_size': batch_size,
                          'n_epochs': n_epochs,
                          'test_name': 'defaultSSV',
                          'device': device or torch.device('cpu'),
                          'losses': [Supervision, ELBo],
                          'loss_params': [[1, 0], [1]],
                          'z_types': [Categorical(supervision_size, 'y_hat', device),
                                      Gaussian(100, 'z', device)],
                          'weight_reconstruction': True}
        kwargs = {**default_kwargs, **kwargs}
        super(DefaultSSVariationalHParams, self).__init__(**kwargs)
        self.target_ignore_index = target_ignore_index
        self.token_ignore_index = token_ignore_index
# ======================================================================================================================
