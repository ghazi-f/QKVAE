# This files serves defining latent variables for PoS tagging


from components.latent_variables import Categorical, Gaussian
from components import links

MARKOVIAN = True


class XInfer(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XInfer, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                     h_params.vocab_ignore_index, markovian=MARKOVIAN)


class XGen(Categorical):
    def __init__(self, h_params, word_embeddings):
        super(XGen, self).__init__(word_embeddings.weight.shape[0], 'x', h_params.device, word_embeddings,
                                   h_params.vocab_ignore_index, markovian=MARKOVIAN)


class ZInfer(Gaussian):
    def __init__(self, h_params):
        super(ZInfer, self).__init__(h_params.z_size, 'z', h_params.device, markovian=MARKOVIAN)


class ZGen(Gaussian):
    def __init__(self, h_params):
        prior_seq_link = links.MLPLink(h_params.z_size, h_params.z_size, h_params.z_size, 1, Gaussian.parameters)
        super(ZGen, self).__init__(h_params.z_size, 'z', h_params.device, prior_sequential_link=prior_seq_link,
                                   markovian=MARKOVIAN, allow_prior=True)


class YInfer(Categorical):
    def __init__(self, h_params, pos_embeddings):
        super(YInfer, self).__init__(pos_embeddings.weight.shape[0], 'y', h_params.device, pos_embeddings,
                                     h_params.pos_ignore_index, markovian=MARKOVIAN)


class YGen(Categorical):
    def __init__(self, h_params, pos_embeddings):
        super(YGen, self).__init__(pos_embeddings.weight.shape[0], 'y', h_params.device, pos_embeddings,
                                   h_params.pos_ignore_index, markovian=MARKOVIAN)

