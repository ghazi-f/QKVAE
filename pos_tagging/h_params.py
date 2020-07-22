# Design this file with a hierarchical experiment checkpointing scheme
import os

from torch import optim
from torch import nn
from components.criteria import *

ROOT_CHECKPOINTING_PATH = 'checkpoints'
ROOT_TENSORBOARD_PATH = 'tb_logs'


# ======================================================================================================================
# ==================================================== HYPER-PARAMETERS ================================================
class DefaultHParams:
    def __init__(self, vocab_size,
                 tag_size,
                 max_len,
                 batch_size,
                 n_epochs,
                 device=None,
                 test_name='default',
                 embedding_dim=300,
                 pos_embedding_dim=20,
                 z_size=500,
                 text_rep_l=1,
                 text_rep_h=400,
                 encoder_h=128,
                 encoder_l=2,
                 pos_h=128,
                 pos_l=2,
                 decoder_h=32,
                 decoder_l=2,
                 losses=None,
                 loss_params=None,
                 piwo=False,
                 optimizer=optim.AdamW,
                 optimizer_kwargs=None,
                 grad_accumulation_steps=1,
                 training_iw_samples=10,
                 testing_iw_samples=100,
                 test_prior_samples=5,
                 is_weighted=None,
                 graph_generator=None,
                 anneal_kl=None,
                 grad_clip=None,
                 kl_th=None,
                 highway=True,
                 dropout=0.,
                 input_dimensions=2,
                 markovian=True,
                 word_dropout=0.0,
                 contiguous_lm=False,
                 n_latents=1):
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
        self.contiguous_lm = contiguous_lm
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.input_dimensions = input_dimensions

        # Architectural hyper-parameters
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.text_rep_l = text_rep_l
        self.text_rep_h = text_rep_h

        self.z_size = z_size

        self.encoder_h = encoder_h
        self.encoder_l = encoder_l
        self.pos_h = pos_h
        self.pos_l = pos_l

        # IDEA: Maybe define z variables through a dictionary to describe tree structured latent variables
        # IDEA: Allow for unallocated bits on the supervised variable

        self.decoder_h = decoder_h
        self.decoder_l = decoder_l

        self.n_latents = n_latents
        self.graph_generator = graph_generator
        self.highway = highway
        self.markovian = markovian

        # Specifying losses
        self.losses = losses or [ELBo]
        self.loss_params = loss_params or [1]*len(self.losses)  # [weight] for unsupervised losses, and [weight, supervised_z_index] for the supervised losses
        self.is_weighted = is_weighted or []
        self.piwo = piwo

        # Optimization hyper-parameters
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3}
        self.grad_accumulation_steps = grad_accumulation_steps
        self.anneal_kl = anneal_kl
        self.grad_clip = grad_clip
        self.kl_th = kl_th
        self.dropout = dropout
        self.word_dropout = word_dropout

        # Sampling hyper_parameters
        self.training_iw_samples = training_iw_samples
        self.testing_iw_samples = testing_iw_samples
        self.test_prior_samples = test_prior_samples

        # This constructing will mainly serve as a sanity-check for the hyper parameter setting
        assert len(self.losses) == len(self.loss_params)
        assert 'lr' in self.optimizer_kwargs


class DefaultSSVariationalHParams(DefaultHParams):
    def __init__(self, vocab_size, tag_size, max_len, batch_size, n_epochs, device=None,
                 pos_ignore_index=None, vocab_ignore_index=None, **kwargs):
        default_kwargs = {'vocab_size': vocab_size,
                          'tag_size': tag_size,
                          'max_len': max_len,
                          'batch_size': batch_size,
                          'n_epochs': n_epochs,
                          'test_name': 'defaultSSV',
                          'device': device or torch.device('cpu'),
                          'losses': [ELBo],#[Supervision, ELBo],
                          'loss_params': [1],#, 1]
                          }
        kwargs = {**default_kwargs, **kwargs}
        super(DefaultSSVariationalHParams, self).__init__(**kwargs)
        self.vocab_ignore_index = vocab_ignore_index
        self.pos_ignore_index = pos_ignore_index


class DefaultSSPoSTagHParams(DefaultHParams):
    def __init__(self, vocab_size, tag_size, max_len, batch_size, n_epochs, device=None,
                 pos_ignore_index=None, vocab_ignore_index=None, **kwargs):
        default_kwargs = {'vocab_size': vocab_size,
                          'tag_size': tag_size,
                          'max_len': max_len,
                          'batch_size': batch_size,
                          'n_epochs': n_epochs,
                          'test_name': 'defaultSSV',
                          'device': device or torch.device('cpu'),
                          'losses': [Supervision, ELBo],
                          }
        kwargs = {**default_kwargs, **kwargs}
        super(DefaultSSPoSTagHParams, self).__init__(**kwargs)
        self.vocab_ignore_index = vocab_ignore_index
        self.pos_ignore_index = pos_ignore_index
# ======================================================================================================================
