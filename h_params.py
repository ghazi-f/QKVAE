# Design this file with a hierarchical experiment checkpointing scheme
import os

from torch import optim
from torch import nn
from components import *

ROOT_CHECKPOINTING_PATH = 'checkpoints'
ROOT_TENSORBOARD_PATH = 'tb_logs'


# ================================================ LATENT VARIABLE TYPE ================================================
class LVTypes:
    # Define the parameters with their corresponding layer activation function
    # NOTE: identity function behaviour can be produced with torch.nn.Sequential() as an activation function
    parameters = {}

    def __init__(self):
        assert len(self.parameters) > 0


class Gaussian(LVTypes):
    parameters = {'m': nn.Sequential, 'sigma': torch.exp}


class Categorical(LVTypes):
    parameters = {'theta': nn.Sequential}


# ======================================================================================================================
# ==================================================== HYPER-PARAMETERS ================================================
class DefaultHParams:
    # A name to be used for checkpoints and Tensorboard logging indexation
    test_name = 'default'
    save_path = os.path.join(ROOT_CHECKPOINTING_PATH, test_name)
    viz_path = os.path.join(ROOT_TENSORBOARD_PATH, test_name)

    # Corpus related hyper-parameters
    vocab_size = None  # TODO: set it right

    # Architectural hyper-parameters
    max_len = 20
    embedding_dim = 100

    encoder = LSTMEncoder
    encoder_h = 32
    encoder_l = 1

    # IDEA: Maybe define z variables through a dictionary to describe tree structured latent variables
    z_sizes = [100]
    z_types = [Gaussian]

    decoder_h = 32
    decoder_l = 1
    decoder = LSTMDecoder

    # Specifying losses
    loss = [Reconstruction]
    loss_weights = [1]

    # Optimization hyper-parameters
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 1e-3}
    grad_accumulation_steps = 1

    def __init__(self):
        # This constructing will mainly serve as a sanity-check for the hyper parameter setting
        assert len(self.loss) == len(self.loss_weights)
        assert 'lr' in self.optimizer_kwargs


class DefaultVariationalHParams(DefaultHParams):
    # A name to be used for checkpoints and Tensorboard logging indexation
    test_name = 'default'
    save_path = os.path.join(ROOT_CHECKPOINTING_PATH, test_name)
    viz_path = os.path.join(ROOT_TENSORBOARD_PATH, test_name)

    # Specifying losses
    loss = [ELBo]
    loss_weights = [1]


class DefaultSSHParams(DefaultHParams):
    # A name to be used for checkpoints and Tensorboard logging indexation
    test_name = 'default'
    save_path = os.path.join(ROOT_CHECKPOINTING_PATH, test_name)
    viz_path = os.path.join(ROOT_TENSORBOARD_PATH, test_name)

    # Specifying losses
    loss = [Supervision, Reconstruction]
    loss_weights = [1, 1]


class DefaultSSVariationalHParams(DefaultHParams):
    # A name to be used for checkpoints and Tensorboard logging indexation
    test_name = 'default'
    save_path = os.path.join(ROOT_CHECKPOINTING_PATH, test_name)
    viz_path = os.path.join(ROOT_TENSORBOARD_PATH, test_name)

    # Specifying losses
    loss = [Supervision, ELBo]
    loss_weights = [1, 1]

# ======================================================================================================================
