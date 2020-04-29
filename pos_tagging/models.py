from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pos_tagging.h_params import *
from components.bayesnets import BayesNet
from components.criteria import Supervision


# ==================================================== BASE MODEL CLASS ================================================

class SSPoSTag(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, tag_index, h_params, autoload=True):
        super(SSPoSTag, self).__init__()

        self.h_params = h_params
        self.word_embeddings = nn.Embedding(h_params.vocab_size, h_params.embedding_dim)
        self.pos_embeddings = nn.Embedding(h_params.tag_size, h_params.pos_embedding_dim)

        # Getting vertices
        vertices, self.supervised_v, self.generated_v = h_params.graph_generator(h_params, self.word_embeddings,
                                                                                 self.pos_embeddings)

        # Instanciating inference and generation networks
        self.infer_bn = BayesNet(vertices['infer'])
        self.gen_bn = BayesNet(vertices['gen'])

        # Setting up categorical variable indexes
        self.index = {self.generated_v: vocab_index, self.supervised_v: tag_index}

        # The losses
        self.losses = [loss(self, w) for loss, w in zip(h_params.losses, h_params.loss_params)]
        self.generate = any([not isinstance(loss, Supervision) for loss in self.losses])
        self.supervise = any([isinstance(loss, Supervision) for loss in self.losses])

        # The Optimizer
        self.optimizer = h_params.optimizer(self.parameters(), **h_params.optimizer_kwargs)
        # Getting the Summary writer
        self.writer = SummaryWriter(h_params.viz_path)
        self.step = 0

        # Loading previous checkpoint if auto_load is set to True
        if autoload:
            self.load()

    def opt_step(self, samples):
        if (self.step % self.h_params.grad_accumulation_steps) == 0:
            # Reinitializing gradients if accumulation is over
            self.optimizer.zero_grad()
        #                          ----------- Unsupervised Forward/Backward ----------------
        # Forward pass
        infer_inputs = {'x': samples['x'][..., 1:]}
        if self.generate:
            self.infer_bn(infer_inputs)
            go_symbol = torch.ones(samples['x'].shape[:-1]).long()*self.index[self.generated_v].stoi['.']
            go_symbol = go_symbol.to(self.h_params.device)
            x_prev = torch.cat([go_symbol.unsqueeze(-1), samples['x'][:, :-1]], dim=-1)
            gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                          **{'x': samples['x'][..., 1:], 'x_prev': samples['x'][..., :-1]}}
            self.gen_bn(gen_inputs)

            # Loss computation and backward pass
            losses_uns = [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]
            sum(losses_uns).backward()
        #                          ------------ supervised Forward/Backward -----------------
        if self.supervised_v.name in samples and self.supervise:
            # Forward pass
            infer_inputs = {**infer_inputs, self.supervised_v.name: samples[self.supervised_v.name]}
            self.infer_bn(infer_inputs)

            # Loss computation and backward pass
            losses_sup = [loss.get_loss() * loss.w for loss in self.losses if isinstance(loss, Supervision)]
            sum(losses_sup).backward()

        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            # Applying gradients if accumulation is over
            self.optimizer.step()
        self.step += 1

        self._dump_train_viz()
        total_loss = (sum(losses_uns) if self.generate else 0) + \
                     (sum(losses_sup) if (self.supervise and self.supervised_v.name in samples) else 0)
        '''for p in self.word_embeddings.parameters():
            print('WE grad', p.grad)

        for p in self.infer_bn.parameters():
            print('BN grad', p.grad)'''
        return total_loss

    def forward(self, samples):
        # Just propagating values through the bayesian networks to get summaries

        #                          ----------- Unsupervised Forward/Backward ----------------
        # Forward pass
        infer_inputs = {'x': samples['x'][..., 1:]}
        if self.generate:
            self.infer_bn(infer_inputs)

            go_symbol = torch.ones(samples['x'].shape[:-1]).long()*self.index[self.generated_v].stoi['<go>']
            go_symbol = go_symbol.to(self.h_params.device)
            x_prev = torch.cat([go_symbol.unsqueeze(-1), samples['x'][:, :-1]], dim=-1)
            gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                          **{'x': samples['x'][..., 1:], 'x_prev': samples['x'][..., :-1]}}
            self.gen_bn(gen_inputs)

            # Loss computation and backward pass
            [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]

        #                          ------------ supervised Forward/Backward -----------------
        if self.supervised_v.name in samples and self.supervise:
            # Forward pass
            infer_inputs = {**infer_inputs, self.supervised_v.name: samples[self.supervised_v.name]}
            self.infer_bn(infer_inputs)

            # Loss computation and backward pass
            [loss.get_loss() * loss.w for loss in self.losses if isinstance(loss, Supervision)]

    def _dump_train_viz(self):
        # Dumping gradient norm
        grad_norm = 0
        for module, name in zip([self, self.infer_bn, self.gen_bn], ['overall', 'inference', 'generation']):
            for p in module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1. / 2)
            self.writer.add_scalar('train' + '/' + '_'.join([name, 'grad_norm']), grad_norm, self.step)

        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for name, metric in loss.metrics().items():
                self.writer.add_scalar('train' + name, metric, self.step)

    def dump_test_viz(self, complete=False):
        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for name, metric in loss.metrics().items():
                self.writer.add_scalar('test' + name, metric, self.step)

        summary_dumpers = {'scalar': self.writer.add_scalar, 'text': self.writer.add_text,
                           'image': self.writer.add_image}

        # We limit the generation of these samples to the less frequent "complete" test visualisations because their
        # computational cost may be high, and because the make the log file a lot larger.
        if complete:
            for summary_type, summary_name, summary_data in self.data_specific_metrics():
                summary_dumpers[summary_type]('test'+summary_name, summary_data, self.step)

    def data_specific_metrics(self):
        # this is supposed to output a list of (summary type, summary name, summary data) triplets
        with torch.no_grad():
            summary_triplets = [
                ('text', '/ground_truth', self.decode_to_text(self.gen_bn.variables_star[self.generated_v])),
                ('text', '/reconstructions', self.decode_to_text(self.gen_bn.variables_hat[self.generated_v])),
            ]

            go_symbol = torch.ones([self.h_params.test_prior_samples]).long() * self.index[self.generated_v].stoi['<go>']
            go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
            x_prev = go_symbol
            for _ in range(self.h_params.max_len):
                self.gen_bn({'x_prev': x_prev})
                x_prev = torch.cat([go_symbol, torch.argmax(self.generated_v.post_params['logits'], dim=-1)], dim=-1)


            '''summary_triplets.append(
                ('text', '/prior_sample', self.decode_to_text(self.gen_bn.variables_hat[self.generated_v])))'''
            summary_triplets.append(
                ('text', '/prior_sample', self.decode_to_text(self.generated_v.post_params['logits'])))

        return summary_triplets

    def decode_to_text(self, x_hat_params):
        # It is assumed that this function is used at test time for display purposes
        # Getting the argmax from the one hot if it's not done
        if x_hat_params.shape[-1] == self.h_params.vocab_size:
            x_hat_params = torch.argmax(x_hat_params, dim=-1)
        text = ' |||| '.join([' '.join([self.index[self.generated_v].itos[x_i_h_p_j]
                                        for x_i_h_p_j in x_i_h_p]).split('<eos>')[0]
                          for x_i_h_p in x_hat_params]).replace('<pad>', '_').replace('<unk>', '<?>')
        return text

    def get_perplexity(self, iterator):
        # TODO: adapt to the new workflow
        with torch.no_grad():
            neg_log_perplexity_lb = []
            total_samples = []
            for batch in tqdm(iterator, desc="Getting Model Perplexity"):
                try:
                    self({'x': batch.text, 'y': batch.label})
                except AttributeError:
                    self({'x': batch.text})
                elbo = -sum([loss.get_loss(actual=True)
                                                   for loss in self.losses if isinstance(loss, ELBo)])
                neg_log_perplexity_lb.append(elbo)

                total_samples.append(torch.sum(batch.text != self.h_params.vocab_ignore_index))

            total_samples = torch.Tensor(total_samples)
            neg_log_perplexity_lb = torch.Tensor(neg_log_perplexity_lb)/torch.sum(total_samples)*total_samples
            neg_log_perplexity_lb = torch.sum(neg_log_perplexity_lb)
            perplexity_ub = torch.exp(- neg_log_perplexity_lb)

            self.writer.add_scalar('test/PerplexityUB', perplexity_ub, self.step)

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

    '''def initialize(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            else:
                nn.init.constant_(param.data, 0)'''


# ======================================================================================================================
# ==================================================== AE MIX-INS ======================================================


# ======================================================================================================================
# ==================================================== MODEL CLASSES ===================================================


# ======================================================================================================================
