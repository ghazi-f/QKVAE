from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

from taln.h_params import *
from components.bayesnets import BayesNet
from components.criteria import Supervision


# ==================================================== SSPOSTAG MODEL CLASS ============================================

class GenerationModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, h_params, autoload=True, wvs=None):
        super(GenerationModel, self).__init__()

        self.h_params = h_params
        self.word_embeddings = nn.Embedding(h_params.vocab_size, h_params.embedding_dim,
                                             padding_idx=vocab_index.stoi['<pad>'])
        nn.init.uniform_(self.word_embeddings.weight, -1., 1.)
        if wvs is not None:
            self.word_embeddings.weight.data.copy_(wvs)
            UNK_IDX = vocab_index.stoi['<unk>']
            PAD_IDX = vocab_index.stoi['<pad>']
            self.word_embeddings.weight.data[UNK_IDX] = torch.zeros(h_params.embedding_dim)
            self.word_embeddings.weight.data[PAD_IDX] = torch.zeros(h_params.embedding_dim)
            #self.word_embeddings.weight.requires_grad = False

        # Getting vertices
        vertices, self.generated_v = h_params.graph_generator(h_params, self.word_embeddings)

        # Instanciating inference and generation networks
        self.infer_bn = BayesNet(vertices['infer'])
        self.infer_last_states = None
        self.infer_last_states_sup = None
        self.gen_bn = BayesNet(vertices['gen'])
        self.gen_last_states = None
        self.gen_last_states_sup = None

        # Setting up categorical variable indexes
        self.index = {self.generated_v: vocab_index}

        # The losses
        self.losses = [loss(self, w) for loss, w in zip(h_params.losses, h_params.loss_params)]
        self.generate = any([not isinstance(loss, Supervision) for loss in self.losses])
        self.iw = any([isinstance(loss, IWLBo) for loss in self.losses])
        if self.iw:
            assert any([lv.iw for lv in self.infer_bn.variables]), "When using IWLBo, at least a variable in the " \
                                                                   "inference graph must be importance weighted."

        # The Optimizer
        self.optimizer = h_params.optimizer(self.parameters(), **h_params.optimizer_kwargs)
        if h_params.cycle_loss_w > 0.0:
            h_params.optimizer_kwargs['lr'] *= h_params.cycle_loss_w
            self.cycle_optimizer = h_params.optimizer(self.gen_bn.parameters(), **h_params.optimizer_kwargs)
            self.cycle_metric = 0
        else:
            self.cycle_metric = None
            self.cycle_optimizer = None
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
        infer_inputs = {'x': samples['x']}

        # Getting sequence lengths
        x_len = (samples['x'] != self.generated_v.ignore).float().sum(-1)
        orig_infer_last_states, orig_gen_last_states = self.infer_last_states, self.gen_last_states
        if self.iw:  # and (self.step >= self.h_params.anneal_kl[0]):
            self.infer_last_states = self.infer_bn(infer_inputs, n_iw=self.h_params.training_iw_samples,
                                                   prev_states=self.infer_last_states, complete=True, lens=x_len)
        else:
            self.infer_last_states = self.infer_bn(infer_inputs, prev_states=self.infer_last_states, complete=True,
                                                   lens=x_len)

        gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                      **{'x': samples['x'], 'x_prev': samples['x_prev']}}
        if self.iw:
            gen_inputs, x_len = self._harmonize_input_shapes(gen_inputs, self.h_params.training_iw_samples, x_len)
        if self.step < self.h_params.anneal_kl[0]:
            self.gen_last_states = self.gen_bn(gen_inputs, target=self.generated_v,
                                               prev_states=self.gen_last_states, lens=x_len)
        else:
            self.gen_last_states = self.gen_bn(gen_inputs, prev_states=self.gen_last_states, lens=x_len)

        # Loss computation and backward pass
        losses_uns = [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]

        # Cleaning computation graph:
        self.gen_bn.clear_values()
        self.infer_bn.clear_values()

        sum(losses_uns).backward()
        if not self.h_params.contiguous_lm:
            self.infer_last_states, self.gen_last_states = None, None

        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            # Applying gradients and gradient clipping if accumulation is over
            if self.h_params.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.h_params.grad_clip)
            self.optimizer.step()
        self.step += 1

        # Calculating cycle loss
        if (self.cycle_optimizer is not None)  : #and (self.step > self.h_params.anneal_kl[0]) :#and (self.step % 5 == 0):
            assert self.h_params.grad_accumulation_steps == 1
            # print("cycle opt at step ", self.step)
            self.cycle_optimizer.zero_grad()
            x_prev = F.one_hot(gen_inputs['x_prev'][..., :1], self.h_params.vocab_size).float()
            z = gen_inputs['z'][..., 0, :].detach()
            seq_len = gen_inputs['z'].shape[-2]+1
            fill_indexes = torch.randint(1, seq_len, [2])
            for i in range(1, seq_len):
                if i in fill_indexes:
                    cycle_gen_inputs = {'z': z.unsqueeze(-2).expand((*z.shape[:-1], i, z.shape[-1])), 'x_prev': x_prev}
                    self.gen_bn(cycle_gen_inputs, target=self.generated_v, prev_states=orig_gen_last_states, complete=True)
                    x_prev = torch.cat([x_prev, self.generated_v.post_samples[..., -1:, :]], -2)
                else:
                    x_prev = torch.cat([x_prev, F.one_hot(infer_inputs['x'][..., i-1:i],
                                                          self.h_params.vocab_size).float()], -2)
            # print(torch.sum(torch.abs(x_prev[..., 1:, :]-F.one_hot(infer_inputs['x'],
            #                                               self.h_params.vocab_size).float())))
            self.infer_bn({'x': x_prev[..., 1:, :]}, prev_states=orig_infer_last_states,
                          complete=True)
            new_z_params = self.infer_bn.name_to_v['z'].post_params
            self.infer_bn.clear_values()
            self.infer_bn(infer_inputs, prev_states=orig_infer_last_states, complete=True, lens=x_len)
            orig_z_params = self.infer_bn.name_to_v['z'].post_params
            sig0, sig1 = new_z_params['scale'] ** 2, orig_z_params['scale'].detach() ** 2
            mu0, mu1 = new_z_params['loc'], orig_z_params['loc'].detach()
            kl = 0.5 * (sig0 / sig1 + (mu1 - mu0) ** 2 / sig1 + torch.log(sig1) - torch.log(sig0) - 1).sum(-1)
            cycle_loss = kl.mean()
            self.cycle_metric = cycle_loss.detach()
            cycle_loss.backward()
            if self.h_params.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.gen_bn.parameters(), self.h_params.grad_clip)
            self.cycle_optimizer.step()

        self._dump_train_viz()
        total_loss = sum(losses_uns)

        return total_loss.item()

    def forward(self, samples, eval=False, prev_states=None, force_iw=None, gen_this=True):
        # Just propagating values through the bayesian networks to get summaries
        if prev_states:
            infer_prev, gen_prev = prev_states
        else:
            infer_prev, gen_prev = None, None

        #                          ----------- Unsupervised Forward/Backward ----------------
        # Forward pass
        infer_inputs = {'x': samples['x'],  'x_prev': samples['x_prev']}
        # Getting sequence lengths
        x_len = (samples['x'] != self.generated_v.ignore).float().sum(-1)
        infer_prev = self.infer_bn(infer_inputs, n_iw=self.h_params.testing_iw_samples, eval=eval,
                                   prev_states=infer_prev, force_iw=force_iw, complete=True, lens=x_len)
        gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                      **{'x': samples['x'], 'x_prev': samples['x_prev']}}
        if self.iw or force_iw:
            gen_inputs, x_len = self._harmonize_input_shapes(gen_inputs, self.h_params.testing_iw_samples, x_len)
        if self.step < self.h_params.anneal_kl[0]:
            gen_prev = self.gen_bn(gen_inputs, target=self.generated_v, eval=eval, prev_states=gen_prev,
                                   complete=True, lens=x_len)
        else:
            gen_prev = self.gen_bn(gen_inputs, eval=eval, prev_states=gen_prev, complete=True, lens=x_len)

        # Loss computation
        [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]

        if self.generate:
            if self.h_params.contiguous_lm:
                return infer_prev, gen_prev
            else:
                return None, None

    def _dump_train_viz(self):
        # Dumping gradient norm
        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps - 1):
            z_gen = [var for var in self.gen_bn.variables if var.name == 'z'][0]
            for module, name in zip([self, self.infer_bn, self.gen_bn,
                                     self.gen_bn.approximator[z_gen] if z_gen in self.gen_bn.approximator else None],
                                    ['overall', 'inference', 'generation', 'prior']):
                if module is None: continue
                grad_norm = 0
                for p in module.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1. / 2)
                self.writer.add_scalar('train' + '/' + '_'.join([name, 'grad_norm']), grad_norm, self.step)
        if self.cycle_optimizer is not None:
            self.writer.add_scalar('train/cycle_metric',self.cycle_metric, self.step)

        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for name, metric in loss.metrics().items():
                self.writer.add_scalar('train' + name, metric, self.step)

    def dump_test_viz(self, complete=False):
        if complete:
            print('Performing complete test')
        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            for name, metric in loss.metrics().items():
                self.writer.add_scalar('test' + name, metric, self.step)

        summary_dumpers = {'scalar': self.writer.add_scalar, 'text': self.writer.add_text,
                           'image': self.writer.add_image}

        # We limit the generation of these samples to the less frequent "complete" test visualisations because their
        # computational cost may be high, and because the make the log file a lot larger.
        if complete and any([isinstance(loss, ELBo) for loss in self.losses]):
            for summary_type, summary_name, summary_data in self.data_specific_metrics():
                summary_dumpers[summary_type]('test'+summary_name, summary_data, self.step)

    def data_specific_metrics(self):
        # this is supposed to output a list of (summary type, summary name, summary data) triplets
        with torch.no_grad():
            summary_triplets = [
                ('text', '/ground_truth', self.decode_to_text(self.gen_bn.variables_star[self.generated_v])),
                ('text', '/reconstructions', self.decode_to_text(self.generated_v.post_params['logits'])),
            ]

            n_samples = self.h_params.n_latents if self.h_params.n_latents > 1 else self.h_params.test_prior_samples
            repeats = 2 if self.h_params.n_latents > 1 else 1
            go_symbol = torch.ones([n_samples*repeats]).long() * self.index[self.generated_v].stoi['<eos>']
            go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
            x_prev = go_symbol
            temp = 1.0
            only_z_sampling = True
            gen_len = self.h_params.max_len * (3 if self.h_params.contiguous_lm else 1)
            z_gen = self.gen_bn.name_to_v['z']
            if z_gen not in self.gen_bn.parent:
                if self.h_params.n_latents > 1:
                    z_sample = []
                    for _ in range(repeats):
                        z_sample_i = z_gen.prior_sample((1,))[0].repeat(n_samples, 1)
                        z_sample_alt = z_gen.prior_sample((1,))[0]
                        for i in range(self.h_params.n_latents):
                            start, end = int(i * self.h_params.z_size / self.h_params.n_latents), \
                                         int((i + 1) * self.h_params.z_size / self.h_params.n_latents)
                            z_sample_i[i, ..., start:end] = z_sample_alt[0, ..., start:end]
                        z_sample.append(z_sample_i)
                    z_sample = torch.cat(z_sample)
                else:
                    z_sample = z_gen.prior_sample((n_samples, ))[0]
            else:
                z_sample = None
            for i in range(gen_len):
                if z_sample is not None:
                    z_input = {'z': z_sample.unsqueeze(1).expand(z_sample.shape[0], i+1, z_sample.shape[1])}
                    self.gen_bn({'x_prev': x_prev, **z_input})
                else:
                    z_sample = None
                    self.gen_bn({'x_prev': x_prev})
                if only_z_sampling:
                    samples_i = self.generated_v.post_params['logits']
                else:
                    samples_i = self.generated_v.posterior(logits=self.generated_v.post_params['logits'],
                                                           temperature=temp).rsample()
                x_prev = torch.cat([x_prev, torch.argmax(samples_i,     dim=-1)[..., -1].unsqueeze(-1)],
                                   dim=-1)

            summary_triplets.append(
                ('text', '/prior_sample', self.decode_to_text(x_prev)))

        return summary_triplets

    def decode_to_text(self, x_hat_params):
        # It is assumed that this function is used at test time for display purposes
        # Getting the argmax from the one hot if it's not done
        while x_hat_params.shape[-1] == self.h_params.vocab_size and x_hat_params.ndim > 3:
            x_hat_params = x_hat_params.mean(0)
        while x_hat_params.ndim > 2 and x_hat_params.shape[-1] != self.h_params.vocab_size:
            x_hat_params = x_hat_params[0]
        if x_hat_params.shape[-1] == self.h_params.vocab_size:
            x_hat_params = torch.argmax(x_hat_params, dim=-1)
        assert x_hat_params.ndim == 2, "Mis-shaped generated sequence: {}".format(x_hat_params.shape)

        text = ' |||| '.join([' '.join([self.index[self.generated_v].itos[w]
                                        for w in sen])#.split('<eos>')[0]
                              for sen in x_hat_params]).replace('<pad>', '_').replace('_unk', '<?>').replace('<eos>', '\n')
        return text

    def get_perplexity(self, iterator):
        with torch.no_grad():
            neg_log_perplexity_lb = 0
            total_samples = 0
            infer_prev, gen_prev = None, None
            force_iw = [v.name for v in self.infer_bn.variables if (v.name in [v.name for v in self.gen_bn.variables]
                                                                    and v in self.infer_bn.parent)]
            iwlbo = IWLBo(self, 1)
            for i, batch in enumerate(tqdm(iterator, desc="Getting Model Perplexity")):
                if batch.text.shape[1] < 2: continue
                infer_prev, gen_prev = self({'x': batch.text[..., 1:],
                                             'x_prev': batch.text[..., :-1]}, prev_states=(infer_prev, gen_prev),
                                            force_iw=force_iw,
                                            )
                if not self.h_params.contiguous_lm:
                    infer_prev, gen_prev = None, None
                elbo = - iwlbo.get_loss(actual=True)
                total_samples_i = torch.sum(batch.text != self.h_params.vocab_ignore_index)
                neg_log_perplexity_lb += elbo * total_samples_i

                total_samples += total_samples_i

            neg_log_perplexity_lb /= total_samples
            perplexity_ub = torch.exp(- neg_log_perplexity_lb)

            self.writer.add_scalar('test/PerplexityUB', perplexity_ub, self.step)
            return perplexity_ub

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

    def reduce_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= factor

    def _harmonize_input_shapes(self, gen_inputs, n_iw, lens):
        # This function repeats inputs to the generation network so that they all have the same shape
        max_n_dims = max([val.ndim + (1 if val.dtype == torch.long else 0) for val in gen_inputs.values()])
        for k, v in gen_inputs.items():
            actual_v_ndim = v.ndim + (1 if v.dtype == torch.long else 0)
            for _ in range(max_n_dims-actual_v_ndim):
                expand_arg = [n_iw]+list(gen_inputs[k].shape)
                gen_inputs[k] = gen_inputs[k].unsqueeze(0).expand(expand_arg)
        if lens is not None:
            for _ in range(max_n_dims-(lens.ndim + 2)):
                expand_arg = [n_iw]+list(lens.shape)
                lens = lens.unsqueeze(0).expand(expand_arg)
            lens = lens.reshape(-1)
        return gen_inputs, lens


# ======================================================================================================================
