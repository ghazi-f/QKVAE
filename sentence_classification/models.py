from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

from sentence_classification.h_params import *
from components.bayesnets import BayesNet
from components.criteria import Supervision


# ==================================================== SSPOSTAG MODEL CLASS ============================================

class SSSentenceClassification(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, tag_index, h_params, autoload=True, wvs=None):
        super(SSSentenceClassification, self).__init__()

        self.h_params = h_params
        self.word_embeddings = nn.Embedding(h_params.vocab_size, h_params.embedding_dim)
        nn.init.uniform_(self.word_embeddings.weight, -1., 1.)
        if wvs is not None:
            self.word_embeddings.weight.data.copy_(wvs)
            self.word_embeddings.weight.requires_grad = False
        self.pos_embeddings = nn.Embedding(h_params.tag_size, h_params.pos_embedding_dim)

        # Getting vertices
        vertices, self.supervised_v, self.generated_v = h_params.graph_generator(h_params, self.word_embeddings,
                                                                                 self.pos_embeddings)

        # Instanciating inference and generation networks
        self.infer_bn = BayesNet(vertices['infer'])
        self.infer_last_states = None
        self.infer_last_states_test = None
        self.gen_bn = BayesNet(vertices['gen'])
        self.gen_last_states = None
        self.gen_last_states_test = None

        # Setting up categorical variable indexes
        self.index = {self.generated_v: vocab_index, self.supervised_v: tag_index}

        # The losses
        self.losses = [loss(self, w) for loss, w in zip(h_params.losses, h_params.loss_params)]
        self.generate = any([not isinstance(loss, Supervision) for loss in self.losses])
        self.iw = any([isinstance(loss, IWLBo) for loss in self.losses])
        if self.iw:
            assert any([lv.iw for lv in self.infer_bn.variables]), "When using IWLBo, at least a variable in the " \
                                                                   "inference graph must be importance weighted."
        self.supervise = any([isinstance(loss, Supervision) for loss in self.losses])
        self.is_supervised_batch = True

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
        self.is_supervised_batch = self.supervised_v.name in samples
        infer_inputs = {'x': samples['x'],  'x_prev': samples['x_prev']}
        if self.generate and not (self.supervised_v.name in samples):
            if self.iw:  # and (self.step >= self.h_params.anneal_kl[0]):
                self.infer_last_states = self.infer_bn(infer_inputs, n_iw=self.h_params.training_iw_samples,
                                                       prev_states=self.infer_last_states, complete=True)
            else:
                self.infer_last_states = self.infer_bn(infer_inputs, prev_states=self.infer_last_states, complete=True)
            gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                          **{'x': samples['x'], 'x_prev': samples['x_prev']}}
            if self.iw:
                gen_inputs = self._harmonize_input_shapes(gen_inputs, self.h_params.training_iw_samples)
            if self.step < self.h_params.anneal_kl[0]:
                self.gen_last_states = self.gen_bn(gen_inputs, target=self.generated_v,
                                                   prev_states=self.gen_last_states)
            else:
                self.gen_last_states = self.gen_bn(gen_inputs, prev_states=self.gen_last_states)

            # Loss computation and backward pass
            losses_uns = [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]
            sum(losses_uns).backward()
            if not self.h_params.contiguous_lm:
                self.infer_last_states, self.gen_last_states = None, None
        #                          ------------ supervised Forward/Backward -----------------
        if self.supervised_v.name in samples and self.supervise:
            # Forward pass
            infer_inputs = {'x': samples['x'],  'x_prev': samples['x_prev'],
                            self.supervised_v.name: samples[self.supervised_v.name]}
            self.infer_bn(infer_inputs, target=self.supervised_v)

            # Loss computation and backward pass
            losses_sup = [loss.get_loss() * loss.w for loss in self.losses if isinstance(loss, Supervision)]
            sum(losses_sup).backward()

        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            # Applying gradients and gradient clipping if accumulation is over
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.h_params.grad_clip)
            self.optimizer.step()
        self.step += 1

        self._dump_train_viz()
        total_loss = (sum(losses_uns) if (self.generate and not(self.supervised_v.name in samples)) else 0) + \
                     (sum(losses_sup) if (self.supervise and self.supervised_v.name in samples) else 0)

        return total_loss

    def forward(self, samples, eval=False, prev_states=None, force_iw=None):
        # Just propagating values through the bayesian networks to get summaries
        if prev_states:
            infer_prev, gen_prev = prev_states
        else:
            infer_prev, gen_prev = None, None

        #                          ----------- Unsupervised Forward/Backward ----------------
        # Forward pass
        self.is_supervised_batch = self.supervised_v.name in samples
        infer_inputs = {'x': samples['x'],  'x_prev': samples['x_prev']}
        if self.generate:
            infer_prev = self.infer_bn(infer_inputs, n_iw=self.h_params.testing_iw_samples, eval=eval,
                                       prev_states=infer_prev, force_iw=force_iw, complete=True)
            gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                          **{'x': samples['x'], 'x_prev': samples['x_prev']}}
            if self.iw or force_iw:
                gen_inputs = self._harmonize_input_shapes(gen_inputs, self.h_params.testing_iw_samples)
            if self.step < self.h_params.anneal_kl[0]:
                gen_prev = self.gen_bn(gen_inputs, target=self.generated_v, eval=eval, prev_states=gen_prev,
                                       complete=True)
            else:
                gen_prev = self.gen_bn(gen_inputs, eval=eval, prev_states=gen_prev, complete=True)

            # Loss computation and backward pass
            [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]

        #                          ------------ supervised Forward/Backward -----------------
        if self.supervised_v.name in samples and self.supervise:
            # Forward pass
            infer_inputs = {'x': samples['x'], 'x_prev': samples['x_prev'],
                            self.supervised_v.name: samples[self.supervised_v.name]}
            self.infer_bn(infer_inputs, target=self.supervised_v, eval=eval)

            # Loss computation and backward pass
            [loss.get_loss() * loss.w for loss in self.losses if isinstance(loss, Supervision)]

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

        # Getting the interesting metrics: this model's loss and some other stuff that would be useful for diagnosis
        for loss in self.losses:
            if not isinstance(loss, Supervision) or self.is_supervised_batch:
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
            go_symbol = torch.ones([n_samples*repeats]).long() * self.index[self.generated_v].stoi['<go>']
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
                else:
                    z_sample = None
                self.gen_bn({'x_prev': x_prev, **z_input})
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
            force_iw = [v.name for v in self.infer_bn.variables if v.name in [v.name for v in self.gen_bn.variables]]
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

    def get_overall_accuracy(self, iterator):
        with torch.no_grad():
            has_supervision = any([isinstance(l, Supervision) for l in self.losses])
            if has_supervision:
                accurate_preds = 0
                total_samples = 0
                for batch in tqdm(iterator, desc="Getting Model overall Accuracy"):
                    self({'x': batch.text[..., 1:], 'x_prev': batch.text[..., :-1], 'y': batch.label}, eval=False)

                    num_classes = self.supervised_v.size
                    predictions = self.supervised_v.post_params['logits'].view(-1, num_classes)
                    target = self.infer_bn.variables_star[self.supervised_v].view(-1)
                    prediction_mask = (target != self.supervised_v.ignore).float()
                    accurate_preds += torch.sum((torch.argmax(predictions, dim=-1) == target).float() * prediction_mask)

                    total_samples += torch.sum(prediction_mask)

                accuracy = accurate_preds/total_samples

                self.writer.add_scalar('test/OverallAccuracy', accuracy, self.step)
                return accuracy
            else:
                print('Model doesn\'t use supervision')
                return self.step/1e6

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

    def _harmonize_input_shapes(self, gen_inputs, n_iw):
        # This function repeats inputs to the generation network so that they all have the same shape
        max_n_dims = max([val.ndim for val in gen_inputs.values()])
        for k, v in gen_inputs.items():
            actual_v_ndim = v.ndim + (1 if v.dtype == torch.long else 0)
            for _ in range(max_n_dims-actual_v_ndim):
                expand_arg = [n_iw]+list(gen_inputs[k].shape)
                gen_inputs[k] = gen_inputs[k].unsqueeze(0).expand(expand_arg)
        return gen_inputs


# ======================================================================================================================
