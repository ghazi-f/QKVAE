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
                                   h_params.vocab_ignore_index, markovian=True)


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
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        size = int(h_params.z_size * h_params.n_latents[index] / max(h_params.n_latents))
        super(ZInferi, self).__init__(size, 'z{}'.format(index+1), h_params.device, markovian=h_params.markovian,
                                     stl=True, iw=iw, repnet=repnet, sequence_lv=True)


class ZInfer1(Gaussian):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        size = int(h_params.z_size * h_params.n_latents[0] / max(h_params.n_latents))
        super(ZInfer1, self).__init__(size, 'z1', h_params.device, markovian=h_params.markovian,
                                     stl=True, iw=iw, repnet=repnet, sequence_lv=True)


class ZInfer2(Gaussian):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        size = int(h_params.z_size * h_params.n_latents[1] / max(h_params.n_latents))
        super(ZInfer2, self).__init__(size, 'z2', h_params.device, markovian=h_params.markovian,
                                     stl=True, iw=iw, repnet=repnet, sequence_lv=True)


class ZInfer3(Gaussian):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        size = int(h_params.z_size * h_params.n_latents[2] / max(h_params.n_latents))
        super(ZInfer3, self).__init__(size, 'z3', h_params.device, markovian=h_params.markovian,
                                     stl=True, iw=iw, repnet=repnet, sequence_lv=True)


class ZGeni(Gaussian):
    def __init__(self, h_params, repnet, index, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[index] / max(h_params.n_latents))
        super(ZGeni, self).__init__(size, 'z{}'.format(index+1), h_params.device,
                                    markovian=h_params.markovian, allow_prior=allow_prior, repnet=repnet,
                                    sequence_lv=True)


class ZGenBari(Gaussian):
    def __init__(self, h_params, repnet, index, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[index] / max(h_params.n_latents))
        super(ZGenBari, self).__init__(size, 'z_bar{}'.format(index+1), h_params.device,
                                       markovian=h_params.markovian, allow_prior=allow_prior, repnet=repnet,
                                       sequence_lv=True, is_placeholder=True)


class ZGen1(Gaussian):
    def __init__(self, h_params, repnet, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[0] / max(h_params.n_latents))
        super(ZGen1, self).__init__(size, 'z1', h_params.device,
                                   markovian=h_params.markovian, allow_prior=allow_prior, repnet=repnet, sequence_lv=True)


class ZGen2(Gaussian):
    def __init__(self, h_params, repnet, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[1] / max(h_params.n_latents))
        super(ZGen2, self).__init__(size, 'z2', h_params.device,
                                   markovian=h_params.markovian, allow_prior=allow_prior, repnet=repnet, sequence_lv=True)


class ZGen3(Gaussian):
    def __init__(self, h_params, repnet, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[2] / max(h_params.n_latents))
        super(ZGen3, self).__init__(size, 'z3', h_params.device,
                                   markovian=h_params.markovian, allow_prior=allow_prior, repnet=repnet, sequence_lv=True)



class ZlstmInfer(Gaussian):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        super(ZlstmInfer, self).__init__(h_params.z_size, 'zlstm', h_params.device, markovian=h_params.markovian,
                                     stl=True, iw=iw, repnet=repnet, sequence_lv=True)


class ZlstmGen(Gaussian):
    def __init__(self, h_params, repnet, allow_prior=False):
        super(ZlstmGen, self).__init__(h_params.z_size, 'zlstm', h_params.device,
                                   markovian=h_params.markovian, allow_prior=allow_prior, repnet=repnet, sequence_lv=True)


class ZInferDisc1(MultiCategorical):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        size = int(h_params.z_size * h_params.n_latents[0] / max(h_params.n_latents))
        embedding = MultiEmbedding(size, h_params.n_latents[0], h_params.z_emb_dim)
        super(ZInferDisc1, self).__init__(size, 'z1', h_params.device, embedding, None, markovian=h_params.markovian,
                                          stl=True, iw=iw, repnet=repnet, sequence_lv=True,
                                          n_disc=h_params.n_latents[0])


class ZInferDisc2(MultiCategorical):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        size = int(h_params.z_size * h_params.n_latents[1] / max(h_params.n_latents))
        embedding = MultiEmbedding(size, h_params.n_latents[1], h_params.z_emb_dim)
        super(ZInferDisc2, self).__init__(size, 'z2', h_params.device, embedding, None, markovian=h_params.markovian,
                                          stl=True, iw=iw, repnet=repnet, sequence_lv=True,
                                          n_disc=h_params.n_latents[1])


class ZInferDisc3(MultiCategorical):
    def __init__(self, h_params, repnet):
        iw = any([l == IWLBo for l in h_params.losses]) and not h_params.ipiwo
        size = int(h_params.z_size * h_params.n_latents[2] / max(h_params.n_latents))
        embedding = MultiEmbedding(size, h_params.n_latents[2], h_params.z_emb_dim)
        super(ZInferDisc3, self).__init__(size, 'z3', h_params.device, embedding, None, markovian=h_params.markovian,
                                     stl=True, iw=iw, repnet=repnet, sequence_lv=True, n_disc=h_params.n_latents[2])


class ZGenDisc1(MultiCategorical):
    def __init__(self, h_params, repnet, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[0] / max(h_params.n_latents))
        embedding = MultiEmbedding(size, h_params.n_latents[0], h_params.z_emb_dim)
        super(ZGenDisc1, self).__init__(size, 'z1', h_params.device, embedding, None, markovian=h_params.markovian,
                                        allow_prior=allow_prior, repnet=repnet, sequence_lv=True,
                                        n_disc=h_params.n_latents[0])


class ZGenDisc2(MultiCategorical):
    def __init__(self, h_params, repnet, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[1] / max(h_params.n_latents))
        embedding = MultiEmbedding(size, h_params.n_latents[1], h_params.z_emb_dim)
        super(ZGenDisc2, self).__init__(size, 'z2', h_params.device, embedding, None, markovian=h_params.markovian,
                                        allow_prior=allow_prior, repnet=repnet, sequence_lv=True,
                                        n_disc=h_params.n_latents[1])


class ZGenDisc3(MultiCategorical):
    def __init__(self, h_params, repnet, allow_prior=False):
        size = int(h_params.z_size * h_params.n_latents[2] / max(h_params.n_latents))
        embedding = MultiEmbedding(size, h_params.n_latents[2], h_params.z_emb_dim)
        super(ZGenDisc3, self).__init__(size, 'z3', h_params.device, embedding, None, markovian=h_params.markovian,
                                        allow_prior=allow_prior, repnet=repnet, sequence_lv=True,
                                        n_disc=h_params.n_latents[2])

