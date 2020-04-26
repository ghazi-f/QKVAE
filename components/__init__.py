

'''# ============================================== BASE CLASSES ==========================================================


class BaseEncoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, h_params, embeddings):
        super(BaseEncoder, self).__init__()
        self.word_embeddings = embeddings

    @abc.abstractmethod
    def forward(self, x):
        # The encoding function
        pass


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, h_params, embeddings):
        super(BaseDecoder, self).__init__()
        self.h_params = h_params
        self.word_embeddings = embeddings

    @abc.abstractmethod
    def forward(self, z, x_prev=None):
        # The decoding function
        pass



# ======================================================================================================================
# ============================================== ENCODER CLASSES =======================================================
class GRUEncoder(BaseEncoder):
    def __init__(self, h_params, embeddings):
        super(GRUEncoder, self).__init__(h_params, embeddings)
        self.rnn = nn.GRU(h_params.embedding_dim, h_params.encoder_h, h_params.encoder_l, batch_first=True)
        self.project_z_prev = nn.Linear(sum([z.size for z in h_params.z_types]), h_params.encoder_h*h_params.encoder_l)

        # Creating a list of modules M that output latent variable parameters
        # Each Mi contains a list of modules Mij so that Mij outputs the parameter j of latent variable i
        self.hidden_to_z_params = nn.ModuleList([nn.ModuleDict({param: nn.Linear(h_params.encoder_h*h_params.encoder_l,
                                                                                 z_type.size)
                                                                for param in z_type.keys()})
                                                 for z_type in h_params.z_types])

    def forward(self, x, z_prev=None):
        embeds = self.word_embeddings(x)
        h_prev = self.project_z_prev(z_prev).view(self.h_params.encoder_l, x.shape[0], self.h_params.encoder_h) \
            if z_prev is not None else None
        rnn_out, h = self.rnn(embeds, hx=h_prev)
        # These parameters are those of q(zi|xi, z<i) for the current word i (even though it's called z_params and
        # not zi_params)
        reshaped_h = h.transpose(0, 1).reshape(x.shape[0], self.h_params.encoder_h*self.h_params.encoder_l)
        z_params = [{param: activation(h2zi_params[param](reshaped_h)) for param, activation in z_type.items()}
                    for h2zi_params, z_type in zip(self.hidden_to_z_params, self.h_params.z_types)]
        return z_params, rnn_out, h



# ======================================================================================================================
# ============================================== DECODER CLASSES========================================================

class GRUDecoder(BaseDecoder):
    def __init__(self, h_params, embeddings):
        super(GRUDecoder, self).__init__(h_params, embeddings)
        self.rnn = nn.GRU(sum([z.size for z in h_params.z_types]), h_params.decoder_h, h_params.decoder_l,
                          batch_first=True)
        self.rnn_to_embedding = torch.nn.Linear(h_params.decoder_h*h_params.decoder_l, h_params.embedding_dim)

        # this converts the concatenation of all z_i to the rnn input
        self.embedding_to_rnn = torch.nn.Linear(h_params.embedding_dim, h_params.decoder_h*h_params.decoder_l)

    def forward(self, z, x_prev=None):
        # The decoding
        h_prev =  self.embedding_to_rnn(x_prev).view(self.h_params.decoder_l, z.shape[0], self.h_params.decoder_h) \
            if x_prev is not None else None
        rnn_out, h = self.rnn(z, hx=h_prev)
        # These parameters are those of p(x_i|z_i, x<i) for the current word i (even though it's called x_params and
        # not xi_params)
        reshaped_h = h.transpose(0, 1).reshape(z.shape[0], self.h_params.decoder_h * self.h_params.decoder_l)
        embedded_rnn_out = self.rnn_to_embedding(reshaped_h)
        x_params = torch.matmul(embedded_rnn_out, self.word_embeddings.weight.transpose(dim0=0, dim1=1))

        return x_params, rnn_out, h


# ======================================================================================================================
# ============================================== CRITERION CLASSES =====================================================


# ======================================================================================================================
'''