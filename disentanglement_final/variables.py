# This files serves defining latent variables for PoS tagging
import torch.nn as nn

from components.latent_variables import Categorical, Gaussian, MultiCategorical, MultiEmbedding
from components import links
from components.criteria import IWLBo


class XInfer(Categorical):
    def __init__(self, h_params, word_embeddings, has_rep=True):
        if has_rep:
            repnet = nn.LSTM(word_embeddings.weight.shape[1], int(h_params.text_rep_h/2),
                             h_params.text_rep_l, batch_first=True, bidirectional=True, dropout=h_params.dropout)
        else:
            repnet = None
        super(XInfer, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                     h_params.vocab_ignore_index, markovian=not has_rep, repnet=repnet)


class XGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XGen, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                   h_params.vocab_ignore_index)


class XPrevGen(Categorical):
    def __init__(self, h_params, word_embeddings, has_rep=True):
        if has_rep:
            repnet = nn.LSTM(word_embeddings.weight.shape[1], h_params.text_rep_h, h_params.text_rep_l,
                             batch_first=True,
                             dropout=h_params.dropout)
        else:
            repnet = None
        super(XPrevGen, self).__init__(word_embeddings.weight.shape[0], 'x_prev', h_params.device, word_embeddings,
                                       h_params.vocab_ignore_index, markovian=not has_rep, is_placeholder=True,
                                       repnet=repnet, word_dropout=h_params.word_dropout)


class ZInferi(Gaussian):
    def __init__(self, h_params, repnet, index):
        size = int(h_params.z_size * h_params.n_latents[index] / max(h_params.n_latents))
        super(ZInferi, self).__init__(size, 'z{}'.format(index+1), h_params.device, stl=True, repnet=repnet,
                                      sequence_lv=True)

class ZpInfer(Gaussian):
    def __init__(self, h_params):
        size = int(h_params.z_size/ max(h_params.n_latents))
        super(ZpInfer, self).__init__(size, 'zp', h_params.device, stl=True, repnet=None, sequence_lv=True)


class ZGeni(Gaussian):
    def __init__(self, h_params, repnet, index, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[index] / max(h_params.n_latents))
        super(ZGeni, self).__init__(size, 'z{}'.format(index+1), h_params.device, allow_prior=allow_prior, repnet=repnet,
                                    sequence_lv=True)

class ZprevGeni(Gaussian):
    def __init__(self, h_params, repnet, index, allow_prior=False):
        size = int(h_params.z_size * (h_params.n_latents[index]-1) / max(h_params.n_latents))
        super(ZprevGeni, self).__init__(size, 'z_prev{}'.format(index+1), h_params.device, allow_prior=allow_prior,
                                        repnet=repnet, sequence_lv=True, is_placeholder=True)

class ZpGen(Gaussian):
    def __init__(self, h_params, allow_prior=True):
        size = int(h_params.z_size/ max(h_params.n_latents))
        super(ZpGen, self).__init__(size, 'zp', h_params.device, allow_prior=allow_prior, repnet=None,
                                    sequence_lv=True)

class GAuxGen(Gaussian):
    def __init__(self, h_params, allow_prior=False):
        size = h_params.z_size
        super(GAuxGen, self).__init__(size, 'g', h_params.device, allow_prior=allow_prior, repnet=None,
                                    sequence_lv=True)

class GPrevAuxGen(Gaussian):
    def __init__(self, h_params, allow_prior=False):
        size = h_params.z_size
        super(GPrevAuxGen, self).__init__(size, 'g_prev', h_params.device, allow_prior=allow_prior, repnet=None,
                                    sequence_lv=True)
class ZStInfer(Gaussian):
    def __init__(self, h_params):
        size = h_params.z_size#int(h_params.z_size/max(h_params.n_latents))
        super(ZStInfer, self).__init__(size, 'zs', h_params.device, stl=True, sequence_lv=True)


class ZStGen(Gaussian):
    def __init__(self, h_params, allow_prior=True):
        size = h_params.z_size#int(h_params.z_size/max(h_params.n_latents))
        super(ZStGen, self).__init__(size, 'zs', h_params.device, allow_prior=allow_prior, sequence_lv=True)


class ZStDiscInfer(Categorical):
    def __init__(self, h_params, embedding):
        size = h_params.z_size
        super(ZStDiscInfer, self).__init__(size, 'zs', h_params.device, embedding, None, stl=True, sequence_lv=True)


class ZStDiscGen(Categorical):
    def __init__(self, h_params, embedding, allow_prior=True):
        size = h_params.z_size
        super(ZStDiscGen, self).__init__(size, 'zs', h_params.device, embedding, None,
                                         allow_prior=allow_prior, sequence_lv=True)


class ZGInfer(Gaussian):
    def __init__(self, h_params):
        size = h_params.z_size#int(h_params.z_size/max(h_params.n_latents))
        super(ZGInfer, self).__init__(size, 'zg', h_params.device, stl=True, sequence_lv=True)


class ZGGen(Gaussian):
    def __init__(self, h_params, allow_prior=True):
        size = h_params.z_size#int(h_params.z_size/max(h_params.n_latents))
        super(ZGGen, self).__init__(size, 'zg', h_params.device, allow_prior=allow_prior,
                                    sequence_lv=True)

# ============================================= LEGACY =============================================


class ZGInferLeg(Gaussian):
    def __init__(self, h_params):
        size = int(h_params.z_size/max(h_params.n_latents))
        super(ZGInferLeg, self).__init__(size, 'zg', h_params.device, stl=True, sequence_lv=True)


class ZGGenLeg(Gaussian):
    def __init__(self, h_params, allow_prior=True):
        size = int(h_params.z_size/max(h_params.n_latents))
        super(ZGGenLeg, self).__init__(size, 'zg', h_params.device, allow_prior=allow_prior, sequence_lv=True)

class ZStInferLeg(Gaussian):
    def __init__(self, h_params):
        size = int(h_params.z_size/max(h_params.n_latents))
        super(ZStInferLeg, self).__init__(size, 'zs', h_params.device, stl=True, sequence_lv=True)


class ZStGenLeg(Gaussian):
    def __init__(self, h_params, allow_prior=True):
        size = int(h_params.z_size/max(h_params.n_latents))
        super(ZStGenLeg, self).__init__(size, 'zs', h_params.device, allow_prior=allow_prior, sequence_lv=True)
