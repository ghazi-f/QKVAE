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
    def __init__(self, c_vocab_size,
                 w_vocab_size,
                 c_max_len,
                 w_max_len,
                 batch_size,
                 n_epochs,
                 device=None,
                 test_name='default',
                 w_embedding_dim=300,
                 c_embedding_dim=30,
                 y_embedding_dim=30,
                 zcom_size=500,
                 zdiff_size=500,
                 c_encoder_h=128,
                 c_encoder_l=2,
                 w_encoder_h=128,
                 w_encoder_l=2,
                 y_encoder_l=2,
                 y_encoder_h=32,
                 c_decoder_h=32,
                 c_decoder_l=2,
                 w_decoder_h=32,
                 w_decoder_l=2,
                 optimizer=optim.AdamW,
                 optimizer_kwargs=None,
                 grad_accumulation_steps=1,
                 testing_iw_samples=100,
                 is_weighted=None,
                 graph_generator=None,
                 anneal_kl=None,
                 grad_clip=None,
                 kl_th=None,
                 dropout=0.,
                 word_dropout=0.0,
                 char_dropout=0.0,
                 contiguous_lm=False,
                 max_elbo=False,
                 n_latents=1,
                 input_dimensions=2):
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
        self.c_vocab_size = c_vocab_size
        self.w_vocab_size = w_vocab_size

        # Architectural hyper-parameters
        self.input_dimensions = input_dimensions
        self.w_max_len = w_max_len
        self.c_max_len = c_max_len
        self.w_embedding_dim = w_embedding_dim
        self.c_embedding_dim = c_embedding_dim
        self.y_embedding_dim = y_embedding_dim

        self.zdiff_size = zdiff_size
        self.zcom_size = zcom_size

        self.c_encoder_h = c_encoder_h
        self.c_encoder_l = c_encoder_l
        self.w_encoder_h = w_encoder_h
        self.w_encoder_l = w_encoder_l
        self.y_encoder_h = y_encoder_h
        self.y_encoder_l = y_encoder_l

        # IDEA: Allow for unallocated bits on the supervised variable

        self.c_decoder_h = c_decoder_h
        self.c_decoder_l = c_decoder_l
        self.w_decoder_h = w_decoder_h
        self.w_decoder_l = w_decoder_l

        self.n_latents = n_latents

        self.graph_generator = graph_generator

        # Specifying losses
        self.is_weighted = is_weighted or []

        # Optimization hyper-parameters
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3}
        self.grad_accumulation_steps = grad_accumulation_steps
        self.anneal_kl = anneal_kl
        self.max_elbo = max_elbo
        self.grad_clip = grad_clip
        self.kl_th = kl_th
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.char_dropout = char_dropout

        # Sampling hyper_parameters
        self.testing_iw_samples = testing_iw_samples

        # This will mainly serve as a sanity-check for the hyper parameter setting
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


class DefaultTransformerHParams(DefaultHParams):
    def __init__(self, c_vocab_size, w_vocab_size, c_max_len, w_max_len, batch_size, n_epochs, device=None,
                 c_ignore_index=None, w_ignore_index=None, y_ignore_index=None, **kwargs):
        default_kwargs = {'c_vocab_size': c_vocab_size,
                          'w_vocab_size': w_vocab_size,
                          'c_max_len': c_max_len,
                          'w_max_len': w_max_len,
                          'batch_size': batch_size,
                          'n_epochs': n_epochs,
                          'test_name': 'defaultSSV',
                          'device': device or torch.device('cpu')
                          }
        kwargs = {**default_kwargs, **kwargs}
        super(DefaultTransformerHParams, self).__init__(**kwargs)
        self.w_ignore_index = w_ignore_index
        self.c_ignore_index = c_ignore_index
        self.y_ignore_index = y_ignore_index
# ======================================================================================================================
