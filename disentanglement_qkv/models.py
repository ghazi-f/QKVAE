import sys

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import SGD
import numpy as np
from tqdm import tqdm
import pandas as pd

from disentanglement_qkv.data_prep import BinaryYelp, BARTParaNMT, BARTYelp, NLIGenData2, BARTNLI
from disentanglement_qkv.h_params import *
from disentanglement_qkv.graphs import get_vanilla_graph
from components.links import CoattentiveTransformerLink, ConditionalCoattentiveTransformerLink, \
    ConditionalCoattentiveQKVTransformerLink, CoattentiveTransformerLink2, ConditionalCoattentiveTransformerLink2, \
    CoattentiveBARTTransformerLink, ConditionalCoattentiveBARTTransformerLink, QKVBartTransformerLink, BartModel
from components.bayesnets import BayesNet
from components.criteria import Supervision
from components.latent_variables import MultiCategorical

# sys.path.insert(0, os.path.join("disentanglement_qkv", "senteval"))
# from senteval.engine import SE

import spacy
# import benepar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from datasets import load_metric
from supar import Parser

BART_GRAPHS = [] #TODO: fill_in graphs

const_parser = Parser.load('crf-con-en')
sns.set_style("ticks", {"xtick.major.color": 'white', "ytick.major.color": 'white'})
bleu_score = load_metric("bleu").compute
nlp = spacy.load("en_core_web_sm")
# try:
#     nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
# except LookupError:
#     benepar.download('benepar_en3')
#     nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))


# ============================================= DISENTANGLEMENT MODEL CLASS ============================================

class DisentanglementTransformerVAE(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, tag_index, h_params, autoload=True, wvs=None, dataset=None):
        self.dataset = dataset
        self.uses_bart = any([isinstance(self.dataset, cl) for cl in [BARTParaNMT, BARTYelp, BARTNLI]])
        self.go_symbol = "<s>" if self.uses_bart else "<go>"
        self.eos_symbol = "</s>" if self.uses_bart else "<eos>"

        super(DisentanglementTransformerVAE, self).__init__()

        self.h_params = h_params
        if self.uses_bart:
            self.word_embeddings = BartModel.from_pretrained('facebook/bart-base').shared
        else:
            self.word_embeddings = nn.Embedding(h_params.vocab_size, h_params.embedding_dim)
            nn.init.uniform_(self.word_embeddings.weight, -1., 1.)
            if wvs is not None:
                self.word_embeddings.weight.data.copy_(wvs)
                #self.word_embeddings.weight.requires_grad = False

        # Getting vertices
        vertices, _, self.generated_v = h_params.graph_generator(h_params, self.word_embeddings)

        # Instanciating inference and generation networks
        self.infer_bn = BayesNet(vertices['infer'])
        self.infer_last_states = None
        self.infer_last_states_test = None
        self.gen_bn = BayesNet(vertices['gen'])
        self.gen_last_states = None
        self.gen_last_states_test = None

        self.has_struct = 'zs' in self.gen_bn.name_to_v
        self.linked_zs = 'zs' in [k.name for k in self.gen_bn.parent.keys()]
        self.has_zg = 'zg' in self.gen_bn.name_to_v

        # Setting up categorical variable indexes
        self.index = {self.generated_v: vocab_index}

        # The losses
        self.losses = [loss(self, w) for loss, w in zip(h_params.losses, h_params.loss_params)]
        self.iw = any([isinstance(loss, IWLBo) for loss in self.losses])
        if self.iw:
            assert any([lv.iw for lv in self.infer_bn.variables]), "When using IWLBo, at least a variable in the " \
                                                                   "inference graph must be importance weighted."

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
        infer_inputs = {'x': samples['x'],  'x_prev': samples['x_prev']}
        alter = np.random.choice(['skip', 'crop'])
        if alter == 'skip':
            shift = np.random.randint(7)
            shifted_x = infer_inputs['x'][..., shift:]
            padding = torch.zeros_like(infer_inputs['x'])[..., :shift]
            infer_inputs['x'] = torch.cat([shifted_x, padding], -1)
        else:
            cropt_at = np.random.randint(12)
            cropped_x = infer_inputs['x'][..., :cropt_at]
            padding = torch.zeros_like(infer_inputs['x'])[..., cropt_at:]
            infer_inputs['x'] = torch.cat([padding, cropped_x], -1)
        if self.iw:  # and (self.step >= self.h_params.anneal_kl[0]):
            self.infer_last_states = self.infer_bn(infer_inputs, n_iw=self.h_params.training_iw_samples,
                                                   prev_states=self.infer_last_states, complete=True)
        else:
            self.infer_last_states = self.infer_bn(infer_inputs, prev_states=self.infer_last_states, complete=True)
        gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                      **{'x': samples['x'], 'x_prev': samples['x_prev']}}
        if self.iw:
            gen_inputs = self._harmonize_input_shapes(gen_inputs, self.h_params.training_iw_samples)
        if self.step < self.h_params.anneal_kl[0] and self.h_params.anneal_kl_type == "linear":
            self.gen_last_states = self.gen_bn(gen_inputs, target=self.generated_v,
                                               prev_states=self.gen_last_states)
        else:
            self.gen_last_states = self.gen_bn(gen_inputs, prev_states=self.gen_last_states)

        # Loss computation and backward pass
        losses_uns = [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]
        sum(losses_uns).backward()
        if not self.h_params.contiguous_lm:
            self.infer_last_states, self.gen_last_states = None, None

        self.infer_bn.clear_values(), self.gen_bn.clear_values()
        torch.cuda.empty_cache()
        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            # Applying gradients and gradient clipping if accumulation is over
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.h_params.grad_clip)
            self.optimizer.step()
        self.step += 1

        self._dump_train_viz()
        total_loss = sum(losses_uns)

        return total_loss

    def forward(self, samples, eval=False, prev_states=None, force_iw=None):
        # Just propagating values through the bayesian networks to get summaries
        if prev_states:
            infer_prev, gen_prev = prev_states
        else:
            infer_prev, gen_prev = None, None

        #                          ----------- Unsupervised Forward ----------------
        # Forward pass
        infer_inputs = {'x': samples['x'],  'x_prev': samples['x_prev']}

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

        # Loss computation
        [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]

        if self.h_params.contiguous_lm:
            return infer_prev, gen_prev
        else:
            return None, None

    def _dump_train_viz(self):
        # Dumping gradient norm
        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps - 1):
            z_gen = [var for var in self.gen_bn.variables if var.name == 'z1'][0]
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
                ('text', '/ground_truth', self.decode_to_text(self.gen_bn.variables_star[self.generated_v]))]
            gen_inputs = {**{lv.name: lv.infer(lv.post_params) for lv in self.infer_bn.variables if lv.name.startswith('z')},
                          **{'x': self.gen_bn.variables_star[self.gen_bn.name_to_v['x_prev']],
                             'x_prev': self.gen_bn.variables_star[self.gen_bn.name_to_v['x_prev']]}}
            self.gen_bn(gen_inputs, target=self.generated_v, complete=True)
            summary_triplets.append(
                ('text', '/reconstructions', self.decode_to_text(self.generated_v.post_params['logits']))
            )

            if self.has_struct:
                zst_gen = self.gen_bn.name_to_v['zs']
            z_gen = self.gen_bn.name_to_v['z1']
            if self.has_zg:
                zg_gen = self.gen_bn.name_to_v['zg']
            n_samples = sum(self.h_params.n_latents) + (1 if self.has_struct else 0)
            repeats = 2
            go_symbol = torch.ones([n_samples*repeats + 2]).long() * self.index[self.generated_v].stoi[self.go_symbol]
            go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
            x_prev = go_symbol
            temp = 1.0
            only_z_sampling = True
            gen_len = self.h_params.max_len * (3 if self.h_params.contiguous_lm else 1)
            # When z_gen and z_s have standard normal priors
            if not self.has_zg:
                # Getting original 2 sentences
                orig_z_sample_1 = z_gen.prior_sample((1,))[0]
                orig_z_sample_2 = z_gen.prior_sample((1,))[0]
                if self.has_struct and not self.linked_zs:
                    orig_zst_sample_1 = zst_gen.prior_sample((1,))[0]
                    orig_zst_sample_2 = zst_gen.prior_sample((1,))[0]
            else:
                orig_zg_sample1 = zg_gen.prior_sample((1,))[0]
                orig_zg_sample2 = zg_gen.prior_sample((1,))[0]
                self.gen_bn({'zg': orig_zg_sample1.unsqueeze(1),
                             'x_prev':torch.zeros((1, 1, self.generated_v.size)).to(self.h_params.device)})
                orig_z_sample_1 = self.gen_bn.name_to_v['z1'].post_samples.squeeze(1)
                if self.has_struct:
                    orig_zst_sample_1 = self.gen_bn.name_to_v['zs'].post_samples.squeeze(1)
                self.gen_bn({'zg': orig_zg_sample2.unsqueeze(1),
                             'x_prev': torch.zeros((1, 1, self.generated_v.size)).to(self.h_params.device)})
                orig_z_sample_2 = self.gen_bn.name_to_v['z1'].post_samples.squeeze(1)
                if self.has_struct:
                    orig_zst_sample_2 = self.gen_bn.name_to_v['zs'].post_samples.squeeze(1)

            child_zs = [self.gen_bn.name_to_v['z{}'.format(i)] for i in range(2, len(self.h_params.n_latents)+1)]
            self.gen_bn({'z1': orig_z_sample_1.unsqueeze(1),
                         **({'zs': orig_zst_sample_1.unsqueeze(1)} if self.has_struct and not self.linked_zs else {}),
                         **({'zg': orig_zg_sample1.unsqueeze(1)} if self.has_zg else {}),
                         'x_prev':torch.zeros((1, 1, self.generated_v.size)).to(self.h_params.device)})
            orig_child_zs_1 = [z.post_samples.squeeze(1) for z in child_zs]
            if self.linked_zs:
                orig_zst_sample_1 = zst_gen.post_samples.squeeze(1)
            if self.has_struct:
                orig_zst_1 = orig_zst_sample_1
            self.gen_bn({'z1': orig_z_sample_2.unsqueeze(1),
                         **({'zs': orig_zst_sample_2.unsqueeze(1)} if self.has_struct and not self.linked_zs else {}),
                         **({'zg': orig_zg_sample2.unsqueeze(1)} if self.has_zg else {}),
                         'x_prev':torch.zeros((1, 1, self.generated_v.size)).to(self.h_params.device)})
            orig_child_zs_2 = [z.post_samples.squeeze(1) for z in child_zs]
            if self.linked_zs:
                orig_zst_sample_2 = zst_gen.post_samples.squeeze(1)
            if self.has_struct:
                orig_zst_2 = orig_zst_sample_2
            # Creating latent variable duplicates
            orig_zs_samples_1 = [orig_z_sample_1] + orig_child_zs_1
            orig_zs_samples_2 = [orig_z_sample_2] + orig_child_zs_2
            zs_samples_1 = [orig_s.repeat(n_samples+1, 1)
                            for orig_s in orig_zs_samples_1]
            zs_samples_2 = [orig_s.repeat(n_samples+1, 1)
                            for orig_s in orig_zs_samples_2]
            if self.has_struct:
                zst_samples_1 = torch.cat([orig_zst_1.repeat(n_samples, 1), orig_zst_2.repeat(1, 1)])
                zst_samples_2 = torch.cat([orig_zst_2.repeat(n_samples, 1), orig_zst_1.repeat(1, 1)])
            # Swapping latent variable values
            offset = 0
            for j in range(len(zs_samples_1)):
                for i in range(1, self.h_params.n_latents[j] + 1):
                    start, end = int((i - 1) * self.h_params.z_size / max(self.h_params.n_latents)), \
                                 int(i * self.h_params.z_size / max(self.h_params.n_latents))
                    zs_samples_1[j][offset + i, ..., start:end] = orig_zs_samples_2[j][0, ..., start:end]
                    zs_samples_2[j][offset + i, ..., start:end] = orig_zs_samples_1[j][0, ..., start:end]
                offset += self.h_params.n_latents[j]

            # Getting output
            z_input = {'z{}'.format(i+1): torch.cat([z_s_i_1, z_s_i_2]).unsqueeze(1)
                       for i, (z_s_i_1, z_s_i_2) in enumerate(zip(zs_samples_1, zs_samples_2))}
            if self.has_struct:
                z_input['zs'] = torch.cat([zst_samples_1, zst_samples_2]).unsqueeze(1)
            if self.has_zg:
                z_input['zg'] = torch.cat([orig_zg_sample1.repeat(n_samples+1, 1),
                                           orig_zg_sample2.repeat(n_samples+1, 1)]).unsqueeze(1)
                # z_input['zg'] = orig_zg_sample.repeat(2*n_samples+2, 1).unsqueeze(1)

            # Normal Autoregressive generation
            x_prev = self.generate_from_z(z_input, x_prev, gen_len=gen_len, only_z_sampling=True, temp=temp)
            # for i in range(gen_len):
            #     self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i+1, v.shape[-1])
            #                                       for k, v in z_input.items()}})
            #     if only_z_sampling:
            #         samples_i = self.generated_v.post_params['logits']
            #     else:
            #         samples_i = self.generated_v.posterior(logits=self.generated_v.post_params['logits'],
            #                                                temperature=temp).rsample()
            #     x_prev = torch.cat([x_prev, torch.argmax(samples_i,     dim=-1)[..., -1].unsqueeze(-1)],
            #                        dim=-1)

            summary_triplets.append(
                ('text', '/prior_sample', self.decode_to_text(x_prev, gen=True)))

        return summary_triplets

    def decode_to_text(self, x_hat_params, gen=False):
        # It is assumed that this function is used at test time for display purposes
        # Getting the argmax from the one hot if it's not done
        while x_hat_params.shape[-1] == self.h_params.vocab_size and x_hat_params.ndim > 3:
            x_hat_params = x_hat_params.mean(0)
        while x_hat_params.ndim > 2 and x_hat_params.shape[-1] != self.h_params.vocab_size:
            x_hat_params = x_hat_params[0]
        if x_hat_params.shape[-1] == self.h_params.vocab_size:
            x_hat_params = torch.argmax(x_hat_params, dim=-1)
        assert x_hat_params.ndim == 2, "Mis-shaped generated sequence: {}".format(x_hat_params.shape)
        if not gen:
            text = ' |||| '.join([self.decode_indices(sen) for sen in x_hat_params]).replace('<pad>', '_').replace('_unk', '<?>')\
                .replace(self.eos_symbol, '\n').replace(' ğ', '')
        else:
            samples = [self.decode_indices(sen) for sen in x_hat_params]
            if not isinstance(self.dataset, NLIGenData2):
                samples = [sen.split(self.eos_symbol)[0] for sen in samples]
            first_sample, second_sample = samples[:int(len(samples)/2)], samples[int(len(samples) / 2):]
            samples = ['**First Sample**\n'] + \
                      [('orig' if i == 0 else 'zs' if i == len(first_sample)-1 else str(i) if sample == first_sample[0]
                       else '**'+str(i)+'**') + ': ' +
                       sample for i, sample in enumerate(first_sample)] + \
                      ['**Second Sample**\n'] + \
                      [('orig' if i == 0 else 'zs' if i == len(second_sample)-1 else str(i) if sample == second_sample[0]
                       else '**'+str(i)+'**') + ': ' +
                       sample for i, sample in enumerate(second_sample)]
            text = ' |||| '.join(samples).replace('<pad>', '_').replace('_unk', '<?>').replace(' ğ', '')

        return text

    def decode_to_text2(self, x_hat_params, vocab_size, vocab_index):
        # It is assumed that this function is used at test time for display purposes
        # Getting the argmax from the one hot if it's not done
        while x_hat_params.shape[-1] == vocab_size and x_hat_params.ndim > 3:
            x_hat_params = x_hat_params.mean(0)
        while x_hat_params.ndim > 2 and x_hat_params.shape[-1] != self.h_params.vocab_size:
            x_hat_params = x_hat_params[0]
        if x_hat_params.shape[-1] == vocab_size:
            x_hat_params = torch.argmax(x_hat_params, dim=-1)
        assert x_hat_params.ndim == 2, "Mis-shaped generated sequence: {}".format(x_hat_params.shape)

        samples = [self.decode_indices(sen) for sen in x_hat_params]

        return samples

    def decode_indices(self, sen):
        if self.uses_bart:
            return self.dataset.tokenizer.decode(sen).split(self.eos_symbol)[0].replace(self.go_symbol, '')
        else:
            return (' '.join([self.index[self.generated_v].itos[w]
                             for w in sen]).replace('!', self.eos_symbol).replace('.', self.eos_symbol).replace('?', self.eos_symbol)
                    .split(self.eos_symbol)[0].replace(self.go_symbol, '').replace('</go>', '').replace(' ğ', '')
                       .replace('<pad>', '_').replace('_unk', '<?>'))

    def get_perplexity(self, iterator):
        with torch.no_grad():
            neg_log_perplexity_lb = 0
            total_samples = 0
            force_iw = ['zg'] if self.has_zg else \
                (['z1'] + (['zs'] if self.has_struct and not self.linked_zs else []))
            iwlbo = IWLBo(self, 1)

            self.gen_bn.clear_values(), self.infer_bn.clear_values()
            # torch.cuda.synchronize(self.h_params.device)
            # torch.cuda.ipc_collect()
            for i, batch in enumerate(tqdm(iterator, desc="Getting Model Perplexity")):
                if batch.text.shape[1] < 2: continue
                self.infer_bn({'x': batch.text[..., 1:]}, n_iw=self.h_params.testing_iw_samples, force_iw=force_iw,
                              complete=True)
                gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                              **{'x': batch.text[..., 1:], 'x_prev': batch.text[..., :-1]}}
                gen_inputs = self._harmonize_input_shapes(gen_inputs, self.h_params.testing_iw_samples)
                self.gen_bn(gen_inputs, complete=True)
                elbo = - iwlbo.get_loss(actual=True)

                batch_size = batch.text.shape[0]
                total_samples_i = torch.sum(batch.text != self.h_params.vocab_ignore_index)
                neg_log_perplexity_lb += elbo * batch_size

                total_samples += total_samples_i
                self.gen_bn.clear_values(), self.infer_bn.clear_values()

            neg_log_perplexity_lb /= total_samples
            perplexity_ub = torch.exp(- neg_log_perplexity_lb)

            self.writer.add_scalar('test/PerplexityUB', perplexity_ub, self.step)
            return perplexity_ub.cpu().detach().item()

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
            actual_v_ndim = v.ndim + (1 if v.dtype == torch.long and
                                      not isinstance(self.gen_bn.name_to_v[k], MultiCategorical) else 0)
            for _ in range(max_n_dims-actual_v_ndim):
                expand_arg = [n_iw]+list(gen_inputs[k].shape)
                gen_inputs[k] = gen_inputs[k].unsqueeze(0).expand(expand_arg)
        return gen_inputs

    def eval(self):
        for v in self.infer_bn.variables | self.gen_bn.variables:
            if v.name != self.generated_v.name and (isinstance(v, MultiCategorical) or isinstance(v, Categorical)):
                v.switch_to_non_relaxed()
        super(DisentanglementTransformerVAE, self).eval()

    def train(self, mode=True):
        for v in self.infer_bn.variables | self.gen_bn.variables:
            if v.name != self.generated_v.name and (isinstance(v, MultiCategorical) or isinstance(v, Categorical)):
                v.switch_to_relaxed()
        super(DisentanglementTransformerVAE, self).train(mode=mode)

    def get_sentiment_summaries(self, iterator):
        with torch.no_grad():
            infer_prev, gen_prev = None, None
            z_vals = [[] for _ in range(sum(self.h_params.n_latents))]
            y_vals = []
            for i, batch in enumerate(tqdm(iterator, desc="Getting Model Sentiment stats")):
                if batch.text.shape[1] < 2: continue
                infer_prev, gen_prev = self({'x': batch.text[..., 1:],
                                             'x_prev': batch.text[..., :-1]}, prev_states=(infer_prev, gen_prev),
                                            )
                y_vals.extend(batch.label[:, 0].cpu().numpy())
                # Getting z values
                z_vals_i = []
                for j, size in enumerate(self.h_params.n_latents):
                    zj_val = self.infer_bn.name_to_v['z{}'.format(j+1)].post_samples
                    for k in range(size):
                        emb_size = int(zj_val.shape[-1]/size)
                        z_vals_i.append(zj_val[..., 0, k*emb_size:(k+1)*emb_size].cpu().numpy())
                for l in range(len(z_vals_i)):
                    z_vals[l].extend(z_vals_i[l])

                if not self.h_params.contiguous_lm:
                    infer_prev, gen_prev = None, None
            print("Collected {} z samples for each of the zs, and {} labels".format([len(v) for v in z_vals], len(y_vals)))
            fold_size = int([len(v) for v in z_vals][0]/2)
            classifier = LogisticRegression()
            auc = []
            for i in tqdm(range(len(z_vals)), desc="Getting latent vector performance"):
                # 2-fold cross-validation
                classifier.fit(z_vals[i][fold_size:], y_vals[fold_size:])
                auc1 = roc_auc_score(y_vals[:fold_size], classifier.predict(z_vals[i][:fold_size]))
                classifier.fit(z_vals[i][:fold_size], y_vals[:fold_size])
                auc2 = roc_auc_score(y_vals[fold_size:], classifier.predict(z_vals[i][fold_size:]))
                auc.append((auc1+auc2)/2)
            print("The produced AUCs were", auc)
            max_auc = max(auc)
            sort_auc_index = np.argsort(auc)
            auc_margin = auc[sort_auc_index[-1]] - auc[sort_auc_index[-2]]
            max_auc_index = sort_auc_index[-1]
            self.writer.add_scalar('test/max_auc', max_auc, self.step)
            self.writer.add_scalar('test/auc_margin', auc_margin, self.step)
            self.writer.add_scalar('test/max_auc_index', max_auc_index, self.step)

            return max_auc, auc_margin, max_auc_index

    def get_paraphrase_bleu(self, iterator, beam_size=5):
        with torch.no_grad():
            orig, para, orig_mod, para_mod, rec = [], [], [], [], []
            zs_infer, z_infer, x_gen = self.infer_bn.name_to_v['zs'], \
                                          {'z{}'.format(i+1):self.infer_bn.name_to_v['z{}'.format(i+1)]
                                           for i in range(len(self.h_params.n_latents))}, self.gen_bn.name_to_v['x']

            go_symbol = torch.ones((1, 1)).long() * self.index[self.generated_v].stoi[self.go_symbol]
            go_symbol = go_symbol.to(self.h_params.device)
            temp = 1.
            for i, batch in enumerate(tqdm(iterator, desc="Getting Model Paraphrase Bleu stats")):
                if batch.text.shape[1] < 2: continue
                # if i > 1: break

                # get source and target sentence latent variable values
                self.infer_bn({'x': batch.text[..., 1:]})
                orig_zs, orig_z = zs_infer.infer(zs_infer.post_params), \
                                           {k: v.post_params['loc'][..., 0, :] for k, v in z_infer.items()}
                self.infer_bn({'x': batch.para[..., 1:]})
                para_zs, para_z = zs_infer.infer(zs_infer.post_params),\
                                           {k: v.post_params['loc'][..., 0, :] for k, v in z_infer.items()}
                if isinstance(zs_infer, Categorical):
                    orig_zs, para_zs = orig_zs[..., 0], para_zs[..., 0]
                else:
                    orig_zs, para_zs = orig_zs[..., 0, :], para_zs[..., 0, :]

                # generate source and target reconstructions with the latent variable swap
                # Inputs: 1) original sentence to be reconstructed,
                #         2) paraphrase with the original's structure
                #         3) original with the paraphrase's structure
                z_input = {'zs': torch.cat([orig_zs, orig_zs, para_zs]).unsqueeze(1),
                           **{k: torch.cat([orig_z[k], para_z[k], orig_z[k]]).unsqueeze(1) for k in para_z.keys()}}
                x_prev = go_symbol.repeat((para_zs.shape[0]*3, 1))
                x_prev = self.generate_from_z2(z_input, x_prev, mask_unk=False, beam_size=beam_size)
                if beam_size > 1:
                    x_prev = x_prev[:int(x_prev.shape[0] / beam_size)]
                # for i in range(self.h_params.max_len):
                #     self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i + 1, v.shape[-1])
                #                                       for k, v in z_input.items()}}, target=x_gen)
                #     samples_i = self.generated_v.post_params['logits']
                #     x_prev = torch.cat([x_prev, torch.argmax(samples_i, dim=-1)[..., -1].unsqueeze(-1)],
                #                        dim=-1)

                # store original sentences, the 2 resulting "paraphrases", and the reconstruction of the original
                text = self.decode_to_text2(x_prev, self.h_params.vocab_size, self.index[self.generated_v])
                rec_i, para_mod_i, orig_mod_i = text[:int(len(text)/3)], text[int(len(text)/3):int(len(text)*2/3)], \
                                                text[int(len(text)*2/3):]
                orig_i = self.decode_to_text2(batch.text[..., 1:], self.h_params.vocab_size, self.index[self.generated_v])
                para_i = self.decode_to_text2(batch.para[..., 1:], self.h_params.vocab_size, self.index[self.generated_v])
                orig.extend([[o.split()] for o in orig_i])
                para.extend([[p.split()] for p in para_i])
                orig_mod.extend([o.split() for o in orig_mod_i])
                para_mod.extend([p.split() for p in para_mod_i])
                rec.extend([r.split() for r in rec_i])
                # for o, r, p, pm, om in zip(orig_i, rec_i, para_i, para_mod_i, orig_mod_i):
                #     print(r==om, o, '|||',  r, '|||',   p, '|||',   pm, '|||',   om)
            # for o, r, pm, om in zip(orig, rec, para_mod, orig_mod):
            #     print([' '.join(o[0]), '|||',  ' '.join(r), '|||',  ' '.join(pm), '|||',  ' '.join(om)])
            # Calculate the 3 bleu scores
            orig_mod_bleu = bleu_score(predictions=orig_mod, references=para)['bleu']*100
            para_mod_bleu = bleu_score(predictions=para_mod, references=orig)['bleu']*100
            rec_bleu = bleu_score(predictions=rec, references=orig)['bleu']*100
            copy_bleu = bleu_score(predictions=[o[0] for o in orig], references=para)['bleu']*100
            print("Copy Bleu :", copy_bleu)

            self.writer.add_scalar('test/orig_mod_bleu', orig_mod_bleu, self.step)
            self.writer.add_scalar('test/para_mod_bleu', para_mod_bleu, self.step)
            self.writer.add_scalar('test/rec_bleu', rec_bleu, self.step)

            return orig_mod_bleu, para_mod_bleu, rec_bleu

    def get_disentanglement_summaries(self):
        df = self._get_stat_data_frame(n_samples=80, n_alterations=10, batch_size=20)
        df.to_csv(os.path.join(self.h_params.viz_path, 'statdf_{}.csv'.format(self.step)))
        # df = pd.read_csv('Autoreg5stats2.csv')
        df['new_rels'] = df['new_rels'].map(revert_to_l1)
        df['rel_diff'] = df['rel_diff'].map(revert_to_l1)
        rel_types = np.unique(np.concatenate(df['new_rels'].array))
        for ty in rel_types:
            concerned = []
            for deps in df['new_rels'].array:
                concerned.append(ty in deps)
            df[ty] = concerned
        d_rel_types = ['d_' + ty for ty in np.unique(np.concatenate(df['rel_diff'].array))]
        for ty in d_rel_types:
            concerned = []
            for deps in df['rel_diff'].array:
                concerned.append(str(ty[2:]) in deps)
            df[ty] = concerned
        grouped = df.groupby('alteration_id')
        # MIG analogous quantity
        dis_diffs1 = 0
        for ty in d_rel_types:
            largest2 = np.array(grouped.mean().nlargest(2, ty)[ty].array)
            dis_diffs1 += largest2[0] - largest2[1]
        dis_diffs2 = 0
        for ty in rel_types:
            largest2 = np.array(grouped.mean().nlargest(2, ty)[ty].array)
            dis_diffs2 += largest2[0] - largest2[1]
        # MIG-sup analogous quantity
        sup_dis_diffs1 = 0
        for i in range(sum(self.h_params.n_latents)):
            try:
                largest2 = np.array(grouped.mean()[d_rel_types].transpose().nlargest(2, i)[i].array)
                sup_dis_diffs1 += largest2[0] - largest2[1]
            except BaseException as e:
                print(d_rel_types, grouped)
                print("/_!_\\ Could not influence any type of relations /_!_\\ ")
                return  0, 0, None, None
        sup_dis_diffs2 = 0
        for i in range(sum(self.h_params.n_latents)):
            largest2 = np.array(grouped.mean()[rel_types].transpose().nlargest(2, i)[i].array)
            sup_dis_diffs2 += largest2[0] - largest2[1]
        plt.figure(figsize=(20, 5))
        img_arr1 = get_hm_array(grouped.mean()[d_rel_types].transpose())
        img_arr2 = get_hm_array(grouped.mean()[rel_types].transpose())
        img_arr1 = torch.from_numpy(img_arr1).permute(2, 0, 1)
        img_arr2 = torch.from_numpy(img_arr2).permute(2, 0, 1)
        self.writer.add_scalar('test/diff_disent', dis_diffs1, self.step)
        self.writer.add_scalar('test/struct_disent', dis_diffs2, self.step)
        self.writer.add_scalar('test/sup_diff_disent', sup_dis_diffs1, self.step)
        self.writer.add_scalar('test/sup_struct_disent', sup_dis_diffs2, self.step)
        self.writer.add_image('test/struct_dis_img', img_arr2, self.step)
        self.writer.add_image('test/diff_dis_img', img_arr1, self.step)
        return dis_diffs1, dis_diffs2, img_arr1, img_arr2

    def _get_alternative_sentences(self, prev_latent_vals, params, var_z_ids, n_samples, gen_len, complete=None):
        h_params = self.h_params

        n_orig_sentences = prev_latent_vals['z1'].shape[0]
        go_symbol = torch.ones([n_samples * n_orig_sentences]).long() * \
                    self.index[self.generated_v].stoi[self.go_symbol]
        go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
        x_prev = go_symbol
        if complete is not None:
            for token in complete.split(' '):
                x_prev = torch.cat(
                    [x_prev, torch.ones([n_samples * n_orig_sentences, 1]).long().to(self.h_params.device) * \
                     self.index[self.generated_v].stoi[token]], dim=1)
            gen_len = gen_len - len(complete.split(' '))

        orig_zs = [prev_latent_vals['z{}'.format(i+1)].repeat(n_samples, 1) for i in range(len(h_params.n_latents))]
        zs = [self.gen_bn.name_to_v['z{}'.format(i+1)] for i in range(len(h_params.n_latents))]
        gen_input = {**{'z{}'.format(i+1): orig_zs[i].unsqueeze(1) for i in range(len(orig_zs))},
                     'x_prev': torch.zeros((n_samples * n_orig_sentences, 1, self.generated_v.size)).to(
                         self.h_params.device)}
        if self.has_struct:
            orig_zst = prev_latent_vals['zs'].repeat(n_samples, 1)
            zst = self.gen_bn.name_to_v['zs']
            gen_input['zs'] = orig_zst.unsqueeze(1)
        if self.has_zg:
            orig_zg = prev_latent_vals['zg'].repeat(n_samples, 1)
            zg = self.gen_bn.name_to_v['zg']
            gen_input['zg'] = zg.prior_sample((n_samples * n_orig_sentences,))[0]
            # gen_input['zg'] = orig_zg.unsqueeze(1)
        self.gen_bn(gen_input)
        if self.has_zg:
            z1_sample = zs[0].posterior_sample(self.gen_bn.name_to_v['z1'].post_params)[0].squeeze(1)
            if self.has_struct:
                zst_sample = zst.posterior_sample(self.gen_bn.name_to_v['zs'].post_params)[0].squeeze(1)
        else:
            z1_sample = zs[0].prior_sample((n_samples * n_orig_sentences,))[0]
            if self.has_struct:
                if self.linked_zs:
                    zst_sample = zst.posterior_sample(self.gen_bn.name_to_v['zs'].post_params)[0].squeeze(1)
                else:
                    zst_sample = zst.prior_sample((n_samples * n_orig_sentences,))[0]
        zs_sample = [z1_sample] +\
                    [z.post_samples.squeeze(1) for z in zs[1:]]

        for id in var_z_ids:
            # id == sum(h_params.n_latents) means its zst
            if id == sum(h_params.n_latents) and self.has_struct:
                orig_zst = zst_sample
            else:
                assert id < sum(h_params.n_latents)
                z_number = sum([id > sum(h_params.n_latents[:i + 1]) for i in range(len(h_params.n_latents))])
                z_index = id - sum(h_params.n_latents[:z_number])
                start, end = int(h_params.z_size / max(h_params.n_latents) * z_index), int(
                    h_params.z_size / max(h_params.n_latents) * (z_index + 1))
                source, destination = zs_sample[z_number], orig_zs[z_number]
                destination[:, start:end] = source[:, start:end]

        z_input = {'z{}'.format(i+1): orig_zs[i].unsqueeze(1) for i in range(len(orig_zs))}
        if self.has_struct:
            z_input['zs'] = orig_zst.unsqueeze(1)
        if self.has_zg:
            z_input['zg'] = orig_zg.unsqueeze(1)

        # Normal Autoregressive generation
        for i in range(gen_len):
            self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i + 1, v.shape[-1])
                                             for k, v in z_input.items()}})
            samples_i = self.generated_v.post_params['logits']

            x_prev = torch.cat([x_prev, torch.argmax(samples_i, dim=-1)[..., -1].unsqueeze(-1)],
                               dim=-1)

        text = self.decode_to_text2(x_prev, self.h_params.vocab_size, self.index[self.generated_v])
        samples = {'z{}'.format(i+1): zs_sample[i].tolist() for i in range(len(orig_zs))}
        if self.has_struct:
            samples['zs'] = zst_sample.tolist()
        if self.has_zg:
            samples['zg'] = orig_zg.tolist()
        return text, samples

    def get_sentences(self, n_samples, gen_len=16, sample_w=False, vary_z=True, complete=None, contains=None,
                      max_tries=100):
        n_latents = self.h_params.n_latents
        final_text, final_samples, final_params = [], {'z{}'.format(i+1): [] for i in range(len(n_latents))},\
                                                      {'z{}'.format(i+1): None for i in range(1, len(n_latents))}
        trys = 0
        while n_samples > 0:
            trys += 1
            go_symbol = torch.ones([n_samples]).long() * \
                        self.index[self.generated_v].stoi[self.go_symbol]
            go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
            x_prev = go_symbol
            if complete is not None:
                for token in complete.split(' '):
                    x_prev = torch.cat([x_prev, torch.ones([n_samples, 1]).long().to(self.h_params.device) * \
                                        self.index[self.generated_v].stoi[token]], dim=1)
                gen_len = gen_len - len(complete.split(' '))
            temp = 1.
            z_gen = self.gen_bn.name_to_v['z1']
            if self.has_struct:
                zst_gen = self.gen_bn.name_to_v['zs']
            if self.has_zg:
                zg_gen = self.gen_bn.name_to_v['zg']
                zg_sample = zg_gen.prior_sample((n_samples,))[0]
                self.gen_bn({'zg': zg_sample.unsqueeze(1),
                             'x_prev':torch.zeros((n_samples, 1, self.generated_v.size)).to(self.h_params.device)})
                z_sample = self.gen_bn.name_to_v['z1'].post_samples.squeeze(1)
                if self.has_struct:
                    zst_sample = self.gen_bn.name_to_v['zs'].post_samples.squeeze(1)
            else:
                z_sample = z_gen.prior_sample((n_samples,))[0]
                if self.has_struct and not self.linked_zs:
                    zst_sample = zst_gen.prior_sample((n_samples,))[0]

            child_zs = [self.gen_bn.name_to_v['z{}'.format(i)] for i in range(2, len(self.h_params.n_latents) + 1)]

            # Structured Z case
            gen_input = {'z1': z_sample.unsqueeze(1),
                        'x_prev': torch.zeros((n_samples, 1, self.generated_v.size)).to(self.h_params.device)}
            if self.has_struct and not self.linked_zs:
                gen_input['zs'] = zst_sample.unsqueeze(1)
            if self.has_zg:
                gen_input['zg'] = zg_sample.unsqueeze(1)
            self.gen_bn(gen_input)
            zs_samples = [z_sample] + [z.post_samples.squeeze(1) for z in child_zs]
            if self.linked_zs:
                zst_sample = self.gen_bn.name_to_v['zs'].post_samples.squeeze(1)
            zs_params = {z.name: z.post_params for z in child_zs}

            z_input = {'z{}'.format(i+1): z_s.unsqueeze(1) for i, z_s in enumerate(zs_samples)}
            if self.has_struct:
                zs_params['zs'] = {k: v.squeeze(1).repeat(n_samples, 1) for k, v in zst_gen.post_params.items()
                                   if k != 'temperature'}
                z_input['zs'] = zst_sample.unsqueeze(1)
            if self.has_zg:
                zs_params['zg'] = {k: v.squeeze(1).repeat(n_samples, 1) for k, v in zg_gen.post_params.items()}
                z_input['zg'] = zg_sample.unsqueeze(1)

            # Normal Autoregressive generation
            for i in range(gen_len):
                self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i + 1, v.shape[-1])
                                                 for k, v in z_input.items()}})
                if not sample_w:
                    samples_i = self.generated_v.post_params['logits']
                else:
                    samples_i = self.generated_v.posterior(logits=self.generated_v.post_params['logits'] / temp,
                                                          temperature=1).rsample()
                x_prev = torch.cat([x_prev, torch.argmax(samples_i, dim=-1)[..., -1].unsqueeze(-1)],
                                   dim=-1)

            text = self.decode_to_text2(x_prev, self.h_params.vocab_size, self.index[self.generated_v])
            if contains is None:
                sample_dic = {'z{}'.format(i+1): z_s for i, z_s in enumerate(zs_samples)}
                if self.has_struct:
                    sample_dic['zs'] = zst_sample
                if self.has_zg:
                    sample_dic['zg'] = zg_sample
                return text, sample_dic, zs_params
            else:
                if self.has_struct:
                    raise NotImplementedError("key word based sampling still not implemented for qkv graph")
                for i in range(n_samples):
                    if any([w in text[i].split(' ') for w in contains]):
                        n_samples -= 1
                        final_text.append(text[i])
                        for z_s, z in zip(zs_samples, [z_gen]+[child_zs]):
                            final_samples[z.name] = z_s
                            if z.name in final_params:
                                if final_params[z.name] is None:
                                    final_params[z.name] = {k: v[i].unsqueeze(0) for k, v in zs_params[z.name].items()}
                                else:
                                    final_params[z.name] = {k: torch.cat([final_params[z.name][k], v[i].unsqueeze(0)])
                                                          for k, v in zs_params[z.name].items()}

            if max_tries == trys:
                raise TimeoutError('Could only find {} sentences containing "{}" in {} samples'
                                   ''.format(len(final_text), contains, n_samples * max_tries))

        final_samples = {k: torch.cat([v_i.unsqueeze(0) for v_i in v]) for k, v in final_samples.items()}
        return final_text, final_samples, final_params

    def _get_stat_data_frame(self, n_samples=50, n_alterations=10, batch_size=25):
        stats = []
        nlatents = self.h_params.n_latents
        # Generating n_samples sentences
        text, samples, params = self.get_sentences(n_samples=n_samples, gen_len=self.h_params.max_len-1, sample_w=False,
                                                   vary_z=True, complete=None)
        orig_rels = batch_sent_relations(text)
        for i in range(int(n_samples / batch_size)):
            for j in tqdm(range(sum(nlatents)), desc="Processing sample {}".format(str(i))):
                # Altering the sentences
                alt_text, _ = self._get_alternative_sentences(
                                                           prev_latent_vals={k: v[i * batch_size:(i + 1) * batch_size]
                                                                             for k, v in samples.items()},
                                                           params=None, var_z_ids=[j], n_samples=n_alterations,
                                                           gen_len=self.h_params.max_len-1, complete=None)
                alt_rels = batch_sent_relations(alt_text)
                # Getting alteration statistics
                for k in range(n_alterations * batch_size):
                    orig_text = text[(i * batch_size) + k % batch_size]
                    try:
                        new_rels, rel_diff = get_sentence_statistics(orig_text, alt_text[k],
                                                                     orig_rels[(i * batch_size) + k % batch_size],
                                                                     alt_rels[k])
                    except RecursionError or IndexError:
                        continue
                    stats.append([orig_text, alt_text[k], j, new_rels, rel_diff])

        header = ['original', 'altered', 'alteration_id', 'new_rels', 'rel_diff']
        df = pd.DataFrame(stats, columns=header)
        return df

    def get_att_and_rel_idx(self, text_in):
        max_len = text_in.shape[-1]
        text_sents = [self.decode_indices(s) for s in text_in]
        # Getting relations' positions
        rel_idx = [out['idx'] for out in shallow_dependencies(text_sents)]
        # Getting layer wise attention values

        CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = True, True
        ConditionalCoattentiveQKVTransformerLink.get_att, CoattentiveTransformerLink2.get_att = True, True
        ConditionalCoattentiveTransformerLink2.get_att, QKVBartTransformerLink.get_att = True, True
        CoattentiveBARTTransformerLink.get_att, ConditionalCoattentiveBARTTransformerLink.get_att = True, True
        self.infer_bn({'x': text_in})
        CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = False, False
        ConditionalCoattentiveQKVTransformerLink.get_att, CoattentiveTransformerLink2.get_att = False, False
        ConditionalCoattentiveTransformerLink2.get_att, QKVBartTransformerLink.get_att = False, False
        CoattentiveBARTTransformerLink.get_att, ConditionalCoattentiveBARTTransformerLink.get_att = False, False
        all_att_weights = []
        for i in range(len(self.h_params.n_latents)):
            trans_mod = self.infer_bn.approximator[self.infer_bn.name_to_v['z{}'.format(i + 1)]]
            all_att_weights.append(trans_mod.att_vals)
        att_weights = []
        for lv in range(sum(self.h_params.n_latents)):
            var_att_weights = []
            lv_layer = sum([lv >= sum(self.h_params.n_latents[:i + 1]) for i in range(len(self.h_params.n_latents))])
            rank = lv - sum(self.h_params.n_latents[:lv_layer])
            for layer_att_vals in all_att_weights[lv_layer]:
                soft_att_vals = layer_att_vals
                att_out = torch.cat([soft_att_vals[:, rank,
                                     :max_len], soft_att_vals[:, rank, max_len:].sum(-1).unsqueeze(-1)]
                                    , -1)
                if lv_layer == 2:
                    att_out[..., -1] *= 0
                var_att_weights.append(att_out.cpu().detach().numpy())
            att_weights.append(var_att_weights)
        # att_vals shape:[sent, lv, layer, tok]
        att_vals = np.transpose(np.array(att_weights), (2, 0, 1, 3)).mean(-2)
        att_maxes = att_vals.argmax(-1).tolist()
        return rel_idx, att_maxes

    def _get_real_sent_argmaxes(self, sents, maxes):
        text = [[self.index[self.generated_v].itos[w] for w in sen] for sen in sents]
        real_indices = torch.zeros_like(sents)
        for i in range(real_indices.shape[0]):
            for j in range(1, real_indices.shape[1]):
                   real_indices[i, j] = real_indices[i, j-1] +(1 if text[i][j].startswith('Ġ')
                                                                 or text[i][j].startswith('<') else 0)
        real_argmaxes = torch.zeros_like(torch.tensor(maxes))
        for i in range(real_argmaxes.shape[0]):
            real_argmaxes[i] = real_indices[i, maxes[i]]
        return real_argmaxes

    def get_encoder_disentanglement_score(self, data_iter):
        rel_idx, att_maxes = [], []
        for i, batch in enumerate(tqdm(data_iter, desc="Getting model relationship accuracy")):
            rel_idx_i, att_maxes_i = self.get_att_and_rel_idx(batch.text[..., 1:])
            if self.uses_bart:
                att_maxes_i = self._get_real_sent_argmaxes(batch.text[..., 1:], att_maxes_i)
            rel_idx.extend(rel_idx_i)
            att_maxes.extend(att_maxes_i)

        lv_scores = []
        for lv in range(sum(self.h_params.n_latents)):
            found = {'subj': [], 'verb': [], 'dobj': [], 'pobj': []}
            for att, rel_pos in zip(att_maxes, rel_idx):
                for k in found.keys():
                    if len(rel_pos[k]):
                        found[k].append(att[lv] in rel_pos[k])
            lv_scores.append(found)
        enc_att_scores = {'subj': [], 'verb': [], 'dobj': [], 'pobj': []}
        for lv in range(sum(self.h_params.n_latents)):
            for k, v in lv_scores[lv].items():
                enc_att_scores[k].append(np.mean(v))

        enc_max_score, enc_disent_score, enc_disent_vars = {}, {}, {}
        for k, v in enc_att_scores.items():
            sort_idx = np.argsort(v)
            enc_disent_vars[k], enc_disent_score[k], enc_max_score[k] = \
                sort_idx[-1], v[sort_idx[-1]] - v[sort_idx[-2]], v[sort_idx[-1]]
        idx_names = ['z{}'.format(i+1) for i in range(sum(self.h_params.n_latents))]

        return pd.DataFrame(enc_att_scores, index=idx_names), enc_max_score, enc_disent_score, enc_disent_vars

    def get_sentence_statistics2(self, orig, sen, orig_relations, alt_relations, orig_temp, alt_temp):
        orig_relations, alt_relations = orig_relations['text'], alt_relations['text']
        same_struct = True
        for k in orig_relations.keys():
            if (orig_relations[k] == '' and alt_relations[k] != '') or (orig_relations[k] == '' and alt_relations[k] != ''):
                same_struct = False

        def get_diff(arg):
            if orig_relations[arg] != '' and alt_relations[arg] != '':
                return orig_relations[arg] != alt_relations[arg], False
            else:
                return False, orig_relations[arg] != alt_relations[arg]
        syn_temp_diff = orig_temp['syn'] != alt_temp['syn']
        lex_temp_diff = orig_temp['lex'] != alt_temp['lex']

        return get_diff('subj'), get_diff('verb'), get_diff('dobj'), get_diff('pobj'), same_struct, \
               syn_temp_diff, lex_temp_diff

    def _get_stat_data_frame2(self, n_samples=2000, n_alterations=1, batch_size=100):
        stats = []
        # Generating n_samples sentences
        text, samples, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                               sample_w=False, vary_z=True, complete=None)
        orig_rels = shallow_dependencies(text)
        orig_temps = truncated_template(text)
        n_lvs = sum(self.h_params.n_latents) +(1 if self.has_struct else 0)
        for _ in tqdm(range(int(n_samples / batch_size)), desc="Generating original sentences"):
            text_i, samples_i, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                                       sample_w=False, vary_z=True, complete=None)
            text.extend(text_i)
            for k in samples.keys():
                samples[k] = torch.cat([samples[k], samples_i[k]])
            orig_rels.extend(shallow_dependencies(text_i))
            orig_temps.extend(truncated_template(text_i))
        for i in range(int(n_samples / batch_size)):
            for j in tqdm(range(n_lvs), desc="Processing sample {}".format(str(i))):
                # Altering the sentences
                alt_text, _ = self._get_alternative_sentences(
                    prev_latent_vals={k: v[i * batch_size:(i + 1) * batch_size]
                                      for k, v in samples.items()},
                    params=None, var_z_ids=[j], n_samples=n_alterations,
                    gen_len=self.h_params.max_len - 1, complete=None)
                alt_rels = shallow_dependencies(alt_text)
                alt_temps = truncated_template(alt_text)
                # Getting alteration statistics
                for k in range(n_alterations * batch_size):
                    orig_text = text[(i * batch_size) + k % batch_size]
                    try:
                        arg0_diff, v_diff, arg1_diff, arg_star_diff, same_struct, syn_temp_diff, lex_temp_diff = \
                            self.get_sentence_statistics2(orig_text, alt_text[k],
                                                          orig_rels[(i * batch_size) + k % batch_size], alt_rels[k],
                                                          orig_temps[(i * batch_size) + k % batch_size], alt_temps[k])
                    except RecursionError or IndexError:
                        continue
                    altered_var = 'z{}'.format(j+1) if j!=(n_lvs-1) else 'zs'
                    stats.append([orig_text, alt_text[k], altered_var, int(arg0_diff[0]), int(v_diff[0]),
                                  int(arg1_diff[0]), int(arg_star_diff[0]), int(arg0_diff[1]), int(v_diff[1]),
                                  int(arg1_diff[1]), int(arg_star_diff[1]), same_struct, syn_temp_diff, lex_temp_diff])

        header = ['original', 'altered', 'alteration_id', 'subj_diff', 'verb_diff', 'dobj_diff', 'pobj_diff',
                  'subj_struct', 'verb_struct', 'dobj_struct', 'pobj_struct', 'same_struct', 'syntemp_diff',
                  'lextemp_diff']
        df = pd.DataFrame(stats, columns=header)
        var_wise_scores = df.groupby('alteration_id').mean()[['subj_diff', 'verb_diff', 'dobj_diff', 'pobj_diff',
                                                              'syntemp_diff', 'lextemp_diff']]
        var_wise_scores_struct = df.groupby('alteration_id').mean()[['subj_struct', 'verb_struct',
                                                                     'dobj_struct', 'pobj_struct']]
        var_wise_scores.set_axis([a.split('_')[0] for a in var_wise_scores.axes[1]], axis=1, inplace=True)
        # renormalizing
        struct_array = np.array(var_wise_scores_struct)
        n_vars = sum(self.h_params.n_latents) + (1 if "zs" in self.gen_bn.name_to_v else 0)
        struct_array = 1-np.concatenate([struct_array, np.zeros((n_vars, 2))], axis=1)
        var_wise_scores = var_wise_scores/struct_array

        disent_score = 0
        lab_wise_disent = {}
        dec_disent_vars = {}
        for lab in ['subj', 'verb', 'dobj', 'pobj']:
            try:
                dec_disent_vars[lab] = var_wise_scores.idxmax()[lab]
            except TypeError :
                dec_disent_vars[lab] = 'none'
                lab_wise_disent[lab] = 0
                continue
            top2 = np.array(var_wise_scores.nlargest(2, lab)[lab])
            diff = top2[0] - top2[1]
            lab_wise_disent[lab] = diff
            disent_score += diff
        for lab in ['syntemp', 'lextemp']:
            try:
                var_wise_scores.idxmax()[lab]
            except TypeError:
                lab_wise_disent[lab] = 0
                continue
            top2 = np.array(var_wise_scores.nlargest(2, lab)[lab])
            diff = top2[0] - top2[1]
            lab_wise_disent[lab] = diff
        return disent_score, lab_wise_disent, var_wise_scores, dec_disent_vars

    def _get_perturbation_tma(self, n_samples=2000, n_alterations=1, batch_size=100):
        stats = []
        assert self.has_struct
        alter_lvs = [list(range(sum(self.h_params.n_latents))), [sum(self.h_params.n_latents)]]
        n_lvs = sum(self.h_params.n_latents) + 1
        # Generating n_samples sentences
        text, samples, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                              sample_w=False, vary_z=True, complete=None)
        for _ in tqdm(range(int(n_samples / batch_size)), desc="Generating original sentences"):
            text_i, samples_i, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                                       sample_w=False, vary_z=True, complete=None)
            text.extend(text_i)
            for k in samples.keys():
                samples[k] = torch.cat([samples[k], samples_i[k]])
        for i in range(int(n_samples / batch_size)):
            for alvs in tqdm(alter_lvs, desc="Processing sample {}".format(str(i))):
                # Altering the sentences
                alt_text, _ = self._get_alternative_sentences(
                    prev_latent_vals={k: v[i * batch_size:(i + 1) * batch_size]
                                      for k, v in samples.items()},
                    params=None, var_z_ids=alvs, n_samples=n_alterations,
                    gen_len=self.h_params.max_len - 1, complete=None)
                # Getting alteration statistics
                orig_texts = [text[(i * batch_size) + k % batch_size] for k in range(n_alterations * batch_size)]
                tma2 = template_match(orig_texts, alt_text, 2)
                tma3 = template_match(orig_texts, alt_text, 3)
                altered_var = 'zc' if alvs[0] != (n_lvs-1) else 'zs'
                for k in range(n_alterations * batch_size):
                    stats.append([orig_texts[k], alt_text[k], altered_var, tma2[k], tma3[k]])

        header = ['original', 'altered', 'alteration_id', 'tma2', 'tma3']
        df = pd.DataFrame(stats, columns=header)
        var_wise_scores = df.groupby('alteration_id').mean()[['tma2', 'tma3']]
        return var_wise_scores

    def get_swap_tma(self, n_samples=2000, batch_size=50, beam_size=2):
        with torch.no_grad():
            assert self.has_struct
            # Generating n_samples sentences
            text, samples, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                                  sample_w=False, vary_z=True, complete=None)
            for _ in tqdm(range(int(n_samples / batch_size) - 1), desc="Generating original sentences"):
                text_i, samples_i, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                                          sample_w=False, vary_z=True, complete=None)
                text.extend(text_i)
                for k in samples.keys():
                    samples[k] = torch.cat([samples[k], samples_i[k]])
            source_sents, target_sents = text[:int(n_samples / 2)], text[int(n_samples / 2):]
            source_lvs, target_lvs = {k: v[:int(n_samples / 2)] for k, v in samples.items()}, \
                                     {k: v[int(n_samples / 2):] for k, v in samples.items()}
            result_sents = []
            inv_result_sents = []
            go_symbol = torch.ones((1, 1)).long() * self.index[self.generated_v].stoi[self.go_symbol]
            go_symbol = go_symbol.to(self.h_params.device)
            temp = 1.
            for i in tqdm(range(int(n_samples / (2 * batch_size))),
                          desc="Getting Model Swap TMA"):
                z_input = {'zs': source_lvs['zs'][i * batch_size:(i + 1) * batch_size].unsqueeze(1),
                           **{'z{}'.format(i + 1): target_lvs['z{}'.format(i + 1)][
                                                   i * batch_size:(i + 1) * batch_size].unsqueeze(1)
                              for i in range(len(self.h_params.n_latents))}}
                inv_z_input = {'zs': target_lvs['zs'][i * batch_size:(i + 1) * batch_size].unsqueeze(1),
                               **{'z{}'.format(i + 1): source_lvs['z{}'.format(i + 1)][
                                                       i * batch_size:(i + 1) * batch_size].unsqueeze(1)
                                  for i in range(len(self.h_params.n_latents))}}
                x_prev = go_symbol.repeat((batch_size, 1))
                x_prev = self.generate_from_z2(z_input, x_prev, mask_unk=False, beam_size=beam_size)
                if beam_size > 1:
                    x_prev = x_prev[:int(x_prev.shape[0] / beam_size)]
                result_sents.extend(self.decode_to_text2(x_prev, self.h_params.vocab_size,
                                                         self.index[self.generated_v]))
                x_prev = go_symbol.repeat((batch_size, 1))
                x_prev = self.generate_from_z2(inv_z_input, x_prev, mask_unk=False, beam_size=beam_size)
                if beam_size > 1:
                    x_prev = x_prev[:int(x_prev.shape[0] / beam_size)]
                inv_result_sents.extend(self.decode_to_text2(x_prev, self.h_params.vocab_size,
                                                             self.index[self.generated_v]))
            test_name = self.h_params.test_name.split("\\")[-1].split("/")[-1]
            dump_location = os.path.join(".data",
                                         "{}_tempdump.tsv".format(test_name))
            with open(dump_location, 'w', encoding="UTF-8") as f:
                for s, t, r, i in zip(source_sents, target_sents, result_sents,
                                      inv_result_sents):
                    f.write('\t'.join([s, t, r, i]) + '\n')

            print("Calculating zs tma...")
            zs_tma2, zs_tma3 = np.mean(template_match(source_sents, result_sents, 2)) * 100, \
                               np.mean(template_match(source_sents, result_sents, 3)) * 100
            print("Calculating zc tma...")
            zc_tma2, zc_tma3 = np.mean(template_match(source_sents, inv_result_sents, 2)) * 100, \
                               np.mean(template_match(source_sents, inv_result_sents, 3)) * 100
            print("Calculating copy tma...")
            copy_tma2, copy_tma3 = np.mean(template_match(source_sents, target_sents, 2)) * 100, \
                                   np.mean(template_match(source_sents, target_sents, 3)) * 100

            print("Calculating bleu scores...")
            zs_bleu = bleu_score(predictions=[s.split() for s in source_sents],
                                 references=[[s.split()] for s in result_sents])['bleu'] * 100
            zc_bleu = bleu_score(predictions=[s.split() for s in source_sents],
                                 references=[[s.split()] for s in inv_result_sents])['bleu'] * 100
            copy_bleu = bleu_score(predictions=[s.split() for s in source_sents],
                                   references=[[s.split()] for s in target_sents])['bleu'] * 100

            self.writer.add_scalar('test/zs_tma2', zs_tma2, self.step)
            self.writer.add_scalar('test/zs_tma3', zs_tma3, self.step)
            self.writer.add_scalar('test/zs_bleu ', zs_bleu, self.step)
            self.writer.add_scalar('test/zc_tma2', zc_tma2, self.step)
            self.writer.add_scalar('test/zc_tma3', zc_tma3, self.step)
            self.writer.add_scalar('test/zc_bleu ', zc_bleu, self.step)
            self.writer.add_scalar('test/copy_tma2', copy_tma2, self.step)
            self.writer.add_scalar('test/copy_tma3', copy_tma3, self.step)
            self.writer.add_scalar('test/copy_bleu ', copy_bleu, self.step)
            scores = {"tma2": {"zs": zs_tma2, "zc": zc_tma2, "copy": copy_tma2},
                      "tma3": {"zs": zs_tma3, "zc": zc_tma3, "copy": copy_tma3},
                      "bleu": {"zs": zs_bleu, "zc": zc_bleu, "copy": copy_bleu}}
            return scores

    def get_disentanglement_summaries2(self, data_iter, n_samples=2000):
        with torch.no_grad():
            if self.h_params.graph_generator != get_vanilla_graph:
                enc_var_wise_scores, enc_max_score, enc_lab_wise_disent, enc_disent_vars = \
                    self.get_encoder_disentanglement_score(data_iter)
                self.writer.add_scalar('test/total_enc_disent_score', sum(enc_lab_wise_disent.values()), self.step)
                for k in enc_lab_wise_disent.keys():
                    self.writer.add_scalar('test/enc_disent_score[{}]'.format(k), enc_lab_wise_disent[k], self.step)
                enc_heatmap = get_hm_array2(enc_var_wise_scores)#, "enc_heatmap_yelp.eps")
                if enc_heatmap is not None:
                    self.writer.add_image('test/encoder_disentanglement', enc_heatmap, self.step)
                encoder_Ndisent_vars = len(set(enc_disent_vars.values()))
                self.writer.add_scalar('test/encoder_Ndisent_vars', encoder_Ndisent_vars, self.step)
            else:
                enc_lab_wise_disent, encoder_Ndisent_vars = {'subj': 0, 'verb': 0, 'dobj': 0, 'pobj': 0}, 0

            dec_disent_score, dec_lab_wise_disent, dec_var_wise_scores, dec_disent_vars\
                = self._get_stat_data_frame2(n_samples=n_samples)
            self.writer.add_scalar('test/total_dec_disent_score', dec_disent_score, self.step)
            for k in dec_lab_wise_disent.keys():
                self.writer.add_scalar('test/dec_disent_score[{}]'.format(k), dec_lab_wise_disent[k], self.step)
            dec_heatmap = get_hm_array2(dec_var_wise_scores)#, "dec_heatmap_yelp.eps")
            if dec_heatmap is not None:
                self.writer.add_image('test/decoder_disentanglement', dec_heatmap, self.step)
            decoder_Ndisent_vars = len(set(dec_disent_vars.values()))
            self.writer.add_scalar('test/decoder_Ndisent_vars', decoder_Ndisent_vars, self.step)
        return dec_lab_wise_disent, enc_lab_wise_disent, decoder_Ndisent_vars, encoder_Ndisent_vars

    def collect_stats(self, data_iter):
        kl, kl_var, rec, mi, nsamples = 0, 0, 0, 0, 0
        infer_prev, gen_prev = None, None
        loss_obj = self.losses[0]
        zs = [(self.infer_bn.name_to_v['z{}'.format(i+1)], self.gen_bn.name_to_v['z{}'.format(i+1)])
              for i in range(len(self.h_params.n_latents))]
        if "zs" in self.gen_bn.name_to_v and isinstance(self.infer_bn.name_to_v['zs'], Gaussian):
            # Mutual information still hasn't been implemented for non Gaussian Latent variables
            zs += [(self.infer_bn.name_to_v['zs'], self.gen_bn.name_to_v['zs'])]
        if "zg" in self.gen_bn.name_to_v:
            zs += [(self.infer_bn.name_to_v['zg'], self.gen_bn.name_to_v['zg'])]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_iter, desc="Getting Model Stats")):
                if batch.text.shape[1] < 2: continue
                infer_prev, gen_prev = self({'x': batch.text[..., 1:],
                                             'x_prev': batch.text[..., :-1]}, prev_states=(infer_prev, gen_prev))
                if not self.h_params.contiguous_lm:
                    infer_prev, gen_prev = None, None
                nsamples += batch.text.shape[0]
                kl += sum([v for k, v in loss_obj.KL_dict.items() if not k.startswith('/Var')]) * batch.text.shape[0]
                kl_var += sum([v**2 for k, v in loss_obj.KL_dict.items() if k.startswith('/Var')]) * batch.text.shape[0]
                rec += loss_obj.log_p_xIz.sum()
                mi += sum([z[0].get_mi(z[1]) for z in zs])
                self.gen_bn.clear_values(), self.infer_bn.clear_values()
        self.writer.add_scalar('test/MI', (mi/nsamples), self.step)
        return (kl/nsamples).cpu().detach().item(), np.sqrt(kl_var/nsamples), \
               - (rec/nsamples).cpu().detach().item(), (mi/nsamples).cpu().detach().item()

    def generate_from_z(self, z_input, x_prev, gen_len=None, only_z_sampling=True, temp=1.0, mask_unk=True):
        unk_mask = torch.ones(x_prev.shape[0], 1, self.h_params.vocab_size).long().to(self.h_params.device)
        if mask_unk:
            unk_mask[..., self.index[self.generated_v].stoi['<unk>']] = 0

        for i in range(gen_len or self.h_params.max_len):
            self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i + 1, v.shape[-1])
                                              for k, v in z_input.items()}})
            unk_mask_i = unk_mask.expand(unk_mask.shape[0], i + 1, unk_mask.shape[-1])
            if only_z_sampling:
                samples_i = self.generated_v.post_params['logits']
            else:
                samples_i = self.generated_v.posterior(logits=self.generated_v.post_params['logits'],
                                                       temperature=temp).rsample()
            x_prev = torch.cat([x_prev, torch.argmax(samples_i*unk_mask_i, dim=-1)[..., -1].unsqueeze(-1)],
                               dim=-1)
        return x_prev

    def generate_from_z2(self, z_input, x_prev, gen_len=None, only_z_sampling=True, temp=1.0, mask_unk=True,
                         beam_size=1):
        eos_idx = (self.index[self.generated_v].stoi["?"], self.index[self.generated_v].stoi["!"],
                   self.index[self.generated_v].stoi["."], self.index[self.generated_v].stoi[self.eos_symbol])
        unk_mask = torch.ones(x_prev.shape[0], 1, self.h_params.vocab_size).long().to(self.h_params.device)
        if mask_unk:
            unk_mask[..., self.index[self.generated_v].stoi['<unk>']] = 0
        ended = [False]*x_prev.shape[0]
        seq_scores = torch.tensor([[0.0]*x_prev.shape[0]]*beam_size).to(x_prev.device)
        if beam_size > 1:
            z_input = {k: v.unsqueeze(0).expand(beam_size, v.shape[0], 1, *v.shape[2:])
                       for k, v in z_input.items()}
            x_prev = x_prev.unsqueeze(0).expand(beam_size, *x_prev.shape)
            unk_mask = unk_mask.unsqueeze(0).expand(beam_size, *unk_mask.shape)
        for i in range(gen_len or self.h_params.max_len):
            if beam_size == 1:
                z_i = {k: v.expand(v.shape[0], i + 1, *v.shape[2:]) for k, v in z_input.items()}
            else:
                z_i = {k: v.expand(beam_size, v.shape[1], i + 1, *v.shape[3:]) for k, v in z_input.items()}
            self.gen_bn({'x_prev': x_prev, **z_i}, target=self.gen_bn.name_to_v['x'])
            unk_mask_i = unk_mask.expand(*unk_mask.shape[:-2], i + 1, unk_mask.shape[-1])
            if only_z_sampling:
                samples_i = self.generated_v.post_params['logits']
            else:
                samples_i = self.generated_v.posterior(logits=self.generated_v.post_params['logits'],
                                                       temperature=temp).rsample()
            if beam_size == 1:
                best_toks = torch.argmax(samples_i*unk_mask_i, dim=-1)
                x_prev = torch.cat([x_prev, best_toks[..., -1].unsqueeze(-1)], dim=-1)
            else:
                next_xprev = torch.zeros((x_prev.shape[0], x_prev.shape[1], x_prev.shape[2]+1)).long().to(x_prev.device)
                for j in range(x_prev.shape[1]):
                    if any([idx in eos_idx for idx in x_prev[0, j]]) or ended[j]:
                        next_xprev[:, j] = torch.cat([x_prev[:, j],
                                                      x_prev[:, j, -1:]*0+eos_idx[-1]], dim=-1)
                        ended[j] = True
                        continue
                    if i==0:
                        sample_ij = samples_i[0, j, -1].reshape(-1)*unk_mask_i[0, j, -1].reshape(-1)
                    else:
                        sample_ij = (samples_i[:, j, -1]+seq_scores[:, j].unsqueeze(-1)).reshape(-1)\
                                    *unk_mask_i[:, j, -1].reshape(-1)

                    tk = torch.topk(sample_ij, k=beam_size, dim=-1)
                    vocab_size = self.h_params.vocab_size
                    b_idx, w_idx = tk.indices.floor_divide(vocab_size), tk.indices % vocab_size
                    seq_scores[:, j] = seq_scores[b_idx, j]+tk.values
                    next_xprev[:, j] = torch.cat([x_prev[b_idx, j], w_idx.unsqueeze(-1)], dim=-1)
                x_prev = next_xprev
        if beam_size > 1:
            x_prev = x_prev.view(x_prev.shape[0]*x_prev.shape[1], x_prev.shape[2])
        return x_prev

    def embed_sents(self, sents):
        with torch.no_grad():
            zs_infer, z_infer, x_gen = self.infer_bn.name_to_v['zs'], \
                                       {'z{}'.format(i + 1): self.infer_bn.name_to_v['z{}'.format(i + 1)]
                                        for i in range(len(self.h_params.n_latents))}, self.gen_bn.name_to_v['x']

            bsz, max_len = len(sents), max([len(s) for s in sents])
            stoi = self.index[self.generated_v].stoi
            inputs = torch.zeros((bsz, max_len)).to(self.h_params.device).long() + stoi['<pad>']
            for i, sen in enumerate(sents):
                for j, tok in enumerate(sen.split()): # This must be changed for model that use more advanced tokenizers
                    inputs[i, j] = stoi[tok] if tok in stoi else stoi['<unk>']

            self.infer_bn({'x': inputs})
            orig_zs, orig_z = zs_infer.rep(zs_infer.infer(zs_infer.post_params))[..., 0, :], \
                              torch.cat([v.post_params['loc'][..., 0, :] for k, v in z_infer.items()], dim=-1)

            return orig_zs, orig_z

    # def get_sent_eval(self):
    #     def prepare(params, samples):
    #         pass
    #
    #     def batcher_zs(params, batch):
    #         batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    #         embeddings = self.embed_sents(batch)[0]
    #         return embeddings.detach().cpu().clone()
    #
    #
    #     def batcher_zc(params, batch):
    #         batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    #         embeddings = self.embed_sents(batch)[1]
    #         return embeddings.detach().cpu().clone()
    #
    #     # Set params for SentEval
    #     print("Performing evaluation with zs")
    #     task_path = os.path.join("disentanglement_qkv", "senteval", "data")
    #     params = {'task_path': task_path, 'usepytorch': True, 'kfold': 10}
    #     params['classifier'] = {'nhid': 50, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    #     se = SE(params, batcher_zs, prepare)
    #
    #     transfer_tasks = [#'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark',
    #                       'BigramShift', 'Depth', 'TopConstituents']
    #
    #     results_zs = se.eval(transfer_tasks)
    #
    #     print("Performing evaluation with zc")
    #     task_path = os.path.join("disentanglement_qkv", "senteval", "data")
    #     params = {'task_path': task_path, 'usepytorch': True, 'kfold': 10}
    #     params['classifier'] = {'nhid': 50, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    #     se = SE(params, batcher_zc, prepare)
    #
    #     transfer_tasks = [#'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark',
    #                       'BigramShift', 'Depth', 'TopConstituents']
    #
    #     results_zc = se.eval(transfer_tasks)
    #     return results_zs, results_zc

    def _get_syn_disent_encoder_hard(self, split="valid", batch_size=100):
        pair_fn = {"valid": os.path.join(".data", "paranmt2", "dev_input.txt"),
                   "test": os.path.join(".data", "paranmt2", "test_input.txt")}[split]
        ref_fn = {"valid": os.path.join(".data", "paranmt2", "dev_ref.txt"),
                  "test": os.path.join(".data", "paranmt2", "test_ref.txt")}[split]
        t1, t2, t3 = [], [], []
        with open(pair_fn, encoding="UTF-8") as f:
            for i, l in enumerate(f):
                if "\t" in l:
                    t1.append(l.split("\t")[0])
                    t2.append(l.split("\t")[1][:-1])
        with open(ref_fn, encoding="UTF-8") as f:
            for i, l in enumerate(f):
                if len(l):
                    t3.append(l[:-1])

        ezs1, ezc1, ezs2, ezc2, ezs3, ezc3 = None, None, None, None, None, None
        for i in range(int(len(t1) / batch_size)):
            ezs1i, ezc1i = self.embed_sents(t1[i * batch_size:(i + 1) * batch_size])
            ezs2i, ezc2i = self.embed_sents(t2[i * batch_size:(i + 1) * batch_size])
            ezs3i, ezc3i = self.embed_sents(t3[i * batch_size:(i + 1) * batch_size])
            if ezs1 is None:
                ezs1, ezc1 = ezs1i, ezc1i
                ezs2, ezc2 = ezs2i, ezc2i
                ezs3, ezc3 = ezs3i, ezc3i
            else:
                ezs1, ezc1 = torch.cat([ezs1, ezs1i]), torch.cat([ezc1, ezc1i])
                ezs2, ezc2 = torch.cat([ezs2, ezs2i]), torch.cat([ezc2, ezc2i])
                ezs3, ezc3 = torch.cat([ezs3, ezs3i]), torch.cat([ezc3, ezc3i])

        s13sims, s23sims = l2_sim(ezs1, ezs3), l2_sim(ezs2, ezs3)
        c13sims, c23sims = l2_sim(ezc1, ezc3), l2_sim(ezc2, ezc3)

        zs_acc = np.mean(s13sims.cpu().detach().numpy() < s23sims.cpu().detach().numpy())
        zc_acc = np.mean(c13sims.cpu().detach().numpy() > c23sims.cpu().detach().numpy())
        print("Paraphrase detection: with zs {}, with zc {}".format(1-zs_acc, zc_acc))
        self.writer.add_scalar('test/hard_zs_enc_acc', zs_acc, self.step)
        self.writer.add_scalar('test/hard_zc_enc_acc', zc_acc, self.step)
        return zs_acc, zc_acc

    def _get_syn_disent_encoder_easy(self, split="valid", batch_size=100):
        template_file = {"valid": os.path.join(".data", "paranmt2", "dev_input.txt"),
                         "test": os.path.join(".data", "paranmt2", "test_input.txt")}[split]
        paraphrase_file = {"valid": os.path.join(".data", "paranmt2", "dev.txt"),
                           "test": os.path.join(".data", "paranmt2", "test.txt")}[split]
        file_names = {"template": template_file, "paraphrase": paraphrase_file}
        accuracies = {"template": {}, "paraphrase": {}}
        for task, file_n in file_names.items():
            t1, t2 = [], []
            with open(file_n, encoding="UTF-8") as f:
                for i, l in enumerate(f):
                    if "\t" in l:
                        t1.append(l.split("\t")[0])
                        t2.append(l.split("\t")[1])

            ezs1, ezc1, ezs2, ezc2 = None, None, None, None
            for i in range(int(len(t1) / batch_size)):
                ezs1i, ezc1i = self.embed_sents(t1[i * batch_size:(i + 1) * batch_size])
                ezs2i, ezc2i = self.embed_sents(t2[i * batch_size:(i + 1) * batch_size])
                if ezs1 is None:
                    ezs1, ezc1 = ezs1i, ezc1i
                    ezs2, ezc2 = ezs2i, ezc2i
                else:
                    ezs1, ezc1 = torch.cat([ezs1, ezs1i]), torch.cat([ezc1, ezc1i])
                    ezs2, ezc2 = torch.cat([ezs2, ezs2i]), torch.cat([ezc2, ezc2i])
            rep_n = 100
            perm_idx = torch.randperm(ezs1.shape[0] * rep_n)
            ezs1, ezc1 = my_repeat(ezs1, rep_n), my_repeat(ezc1, rep_n)
            ezs2, ezc2 = my_repeat(ezs2, rep_n), my_repeat(ezc2, rep_n)
            ezs3, ezc3 = ezs1[perm_idx], ezc1[perm_idx]

            s12sims, s13sims = l2_sim(ezs1, ezs2), l2_sim(ezs1, ezs3)
            c12sims, c13sims = l2_sim(ezc1, ezc2), l2_sim(ezc1, ezc3)
            syn_emb_sc = np.mean(s12sims.cpu().detach().numpy() > s13sims.cpu().detach().numpy())
            cont_emb_sc = np.mean(c12sims.cpu().detach().numpy() > c13sims.cpu().detach().numpy())
            accuracies[task] = {"zs": syn_emb_sc, "zc": cont_emb_sc}
        print("Paraphrase results 1 : ", accuracies)
        self.writer.add_scalar('test/zs_enc_para_acc', accuracies["paraphrase"]["zs"], self.step)
        self.writer.add_scalar('test/zc_enc_para_acc', accuracies["paraphrase"]["zc"], self.step)
        self.writer.add_scalar('test/zs_enc_temp_acc', accuracies["template"]["zs"], self.step)
        self.writer.add_scalar('test/zc_enc_temp_acc', accuracies["template"]["zc"], self.step)
        return accuracies

    def get_syn_disent_encoder(self, split="valid", batch_size=100):
        easy_scores = self._get_syn_disent_encoder_easy(split=split, batch_size=batch_size)
        hard_zs_score, hard_zs_score = self._get_syn_disent_encoder_hard(split=split, batch_size=batch_size)
        scores = {**easy_scores, "hard": {"zs": hard_zs_score, "zc": hard_zs_score}}
        return scores


class LaggingDisentanglementTransformerVAE(DisentanglementTransformerVAE, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, tag_index, h_params, autoload=True, wvs=None, dataset=None, enc_iter=None):
        self.dataset = dataset
        super(LaggingDisentanglementTransformerVAE, self).__init__(vocab_index, tag_index, h_params,
                                                                   autoload=False, wvs=wvs, dataset=dataset)
        # Encoder iterator
        self._enc_iter = enc_iter
        self.enc_iter = iter(enc_iter)

        # The Optimizer
        self.optimizer = None
        self.aggressive = True
        self.inf_optimizer = h_params.optimizer(self.infer_bn.parameters(), **h_params.optimizer_kwargs)
        self.gen_optimizer = h_params.optimizer(self.gen_bn.parameters(), **h_params.optimizer_kwargs)#SGD(self.gen_bn.parameters(), lr=1.)

        # Loading previous checkpoint if auto_load is set to True
        if autoload:
            self.load()

    def opt_step(self, samples):
        if self.aggressive:
            prev_loss = 10**10
            curr_loss = 0
            burn_log_steps = 4
            n_words = 0
            for i in range(1, 24):
                curr_loss += self._opt_step(None, mode="encoder")
                n_words += self.losses[0].valid_n_samples
                if i % burn_log_steps == 0:
                    curr_loss /= n_words
                    if curr_loss > prev_loss:
                        break
                    else:
                        prev_loss = curr_loss
                        curr_loss = 0
                        n_words = 0
            return self._opt_step(samples, "decoder")
        else:
            return self._opt_step(samples, "both")

    def _opt_step(self, samples, mode="both"):
        if mode == "encoder":
            opt_encoder, opt_decoder = True, False
            try:
                batch = next(self.enc_iter)
            except StopIteration:
                self.enc_iter = iter(self._enc_iter)
                batch = next(self.enc_iter)
            samples = {'x': batch.text[..., 1:], 'x_prev': batch.text[..., :-1]}
        elif mode == "decoder":
            opt_encoder, opt_decoder = False, True
        elif mode == "both":
            opt_encoder, opt_decoder = True, True
        else:
            raise NotImplementedError("unrecognized mode : {}".format(mode))

        self.inf_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        #                          ----------- Unsupervised Forward/Backward ----------------
        # Forward pass
        infer_inputs = {'x': samples['x'], 'x_prev': samples['x_prev']}
        alter = np.random.choice(['skip', 'crop'])
        if alter == 'skip':
            shift = np.random.randint(7)
            shifted_x = infer_inputs['x'][..., shift:]
            padding = torch.zeros_like(infer_inputs['x'])[..., :shift]
            infer_inputs['x'] = torch.cat([shifted_x, padding], -1)
        else:
            cropt_at = np.random.randint(12)
            cropped_x = infer_inputs['x'][..., :cropt_at]
            padding = torch.zeros_like(infer_inputs['x'])[..., cropt_at:]
            infer_inputs['x'] = torch.cat([padding, cropped_x], -1)
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


        if opt_encoder:
            torch.nn.utils.clip_grad_norm_(self.infer_bn.parameters(), self.h_params.grad_clip)# 1.)
            self.inf_optimizer.step()
        if opt_decoder:
            torch.nn.utils.clip_grad_norm_(self.gen_bn.parameters(), self.h_params.grad_clip)#0.5)
            self.gen_optimizer.step()
            self.step += 1

            self._dump_train_viz()
        total_loss = sum(losses_uns)

        return total_loss

    def _dump_train_viz(self):
        # Dumping gradient norm
        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps - 1):
            z_gen = [var for var in self.gen_bn.variables if var.name == 'z1'][0]
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
            for name, metric in loss.metrics().items():
                self.writer.add_scalar('train' + name, metric, self.step)

    def save(self):
        root = ''
        for subfolder in self.h_params.save_path.split(os.sep)[:-1]:
            root = os.path.join(root, subfolder)
            if not os.path.exists(root):
                os.mkdir(root)
        torch.save({'model_checkpoint': self.state_dict(), 'step': self.step, 'aggr':self.aggressive},
                   self.h_params.save_path)
        print("Model {} saved !".format(self.h_params.test_name))

    def load(self):
        if os.path.exists(self.h_params.save_path):
            checkpoint = torch.load(self.h_params.save_path)
            model_checkpoint, self.step, self.aggressive = checkpoint['model_checkpoint'], checkpoint['step'], \
                                                           checkpoint['aggr']
            self.load_state_dict(model_checkpoint)
            print("Loaded model at step", self.step)
        else:
            print("Save file doesn't exist, the model will be trained from scratch.")

# =========================================== DISENTANGLEMENT UTILITIES ================================================
# def batch_sent_relations(sents):
#     target = [{'sentence': sent} for sent in sents]
#     preds = predictor.predict_batch_json(target)
#     sent_dicts = []
#     for pred in preds:
#         sent_dict = []
#         for el in pred['verbs']:
#             sent_dict.append({})
#             for v_i in el['description'].split('[')[1:]:
#                 in_bracket = v_i.split(']')[0]
#                 try:
#                     arg_l, arg_str = in_bracket.split(':')
#                     sent_dict[-1][arg_l] = arg_str
#                 except ValueError as e:
#                     print('this raised an anomaly:', el)
#         sent_dicts.append(sent_dict)
#     return sent_dicts


def get_sentence_statistics(orig, sen, orig_relations=None, relations=None):
    # print(orig, sen)
    orig, sen = orig.replace('<?>', 'UNK'), sen.replace('<?>', 'UNK')
    # Orig properties
    orig_relations = orig_relations
    orig_rel_labs = list(orig_relations[0].keys()) if len(orig_relations) else []

    # Alt properties
    relations = relations
    rel_labs = list(relations[0].keys()) if len(relations) else []
    # Differences
    new_rels = np.union1d(np.setdiff1d(orig_rel_labs, rel_labs), np.setdiff1d(rel_labs, orig_rel_labs)).tolist()

    if len(new_rels) or len(rel_labs) == 0 or len(orig_rel_labs) == 0:
        rel_diff = []
    else:
        rel_diff = [k for k, v in orig_relations[0].items() if orig_relations[0][k] != relations[0][k]]
    return new_rels, rel_diff


def get_hm_array(df):
    snsplt = sns.heatmap(df, cmap='RdYlGn', linewidths=0.20, annot=False)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.gcf().canvas.draw()
    fig_shape = plt.gcf().get_size_inches()*plt.gcf().dpi
    fig_shape = (int(fig_shape[1]), int(fig_shape[0]), 3)
    img_arr = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig_shape)
    return img_arr

def get_hm_array2(df, save_as=None):
    plt.clf()
    # df = df[['subj', 'verb', 'dobj', 'pobj']]
    try:
        snsplt = sns.heatmap(df, cmap ='Reds', linewidths = 0.20, annot=True)
    except TypeError:
        return None
    if save_as is not None:
        snsplt.get_figure().savefig(save_as)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.gcf().canvas.draw()
    fig_shape = plt.gcf().get_size_inches()*plt.gcf().dpi
    fig_shape = (int(fig_shape[1]), int(fig_shape[0]), 3)
    img_arr = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig_shape)
    img_arr = torch.from_numpy(img_arr).permute(2, 0, 1)
    return img_arr


def revert_to_l1(el):
    if type(el) == list or type(el) == np.ndarray:
        return el
    if len(el[1:-1]):
        el = el.replace('(', '').replace("'", '').replace(' ', '').replace('),', ')').replace(']', '').replace('[', '')
        output = [el_i.split(",") for el_i in el.split(')') if len(el_i)>3]
        if len(output)>1:
            output = np.concatenate(output)
        return np.unique(output)
    else:
        return []


def shallow_dependencies(sents):
    docs = nlp.pipe(sents)
    relations = []
    for doc in docs:
        subj, verb, dobj, pobj = ['', []], ['', []], ['', []], ['', []]
        for i, tok in enumerate(doc):
            if tok.dep_ =='ROOT' and tok.pos_ == 'VERB':
                verb = [tok.text, [tok.i]]
            if tok.dep_ == 'nsubj' and subj[0] == '':
                subj = [' '.join([toki.text for toki in tok.subtree]), [toki.i for toki in tok.subtree]]
            if tok.dep_ == 'dobj' and dobj[0] == '':
                dobj = [' '.join([toki.text for toki in tok.subtree]), [toki.i for toki in tok.subtree]]
            if tok.dep_ == 'pobj' and pobj[0] == '':
                pobj = [' '.join([toki.text for toki in tok.subtree]), [toki.i for toki in tok.subtree]]
        relations.append({'text':{'subj': subj[0], 'verb': verb[0], 'dobj': dobj[0], 'pobj': pobj[0]},
                         'idx':{'subj': subj[1], 'verb': verb[1], 'dobj': dobj[1], 'pobj': pobj[1]}})
    return relations


def get_children(tok, depth):
    if depth == 0:
        return list(tok.children)
    else:
        return list(tok.children) + \
               list(itertools.chain.from_iterable([get_children(c, depth-1) for c in tok.children]))


def truncated_template(sents, depth=0):
    docs = nlp.pipe(sents)
    templates = []
    for doc in docs:
        children = None
        for i, tok in enumerate(doc):
            if tok.dep_ =='ROOT':
                children = [tok]+get_children(tok, depth)
        if children is not None:
            sort_dict_lex = {c.i: c.text for c in children}
            sort_dict_syn = {c.i: c.dep_ for c in children}
            templates.append({'lex': ' '.join([sort_dict_lex[i] for i in sorted(sort_dict_lex.keys())]),
                              'syn': ' '.join([sort_dict_syn[i] for i in sorted(sort_dict_syn.keys())])})
        else:
            templates.append({'lex': ' ', 'syn': ' '})
    return templates


def truncate_tree(tree, lv):
    tok_i = 0
    curr_lv = 0
    tree_toks = tree.split()
    while tok_i != len(tree_toks):
        if tree_toks[tok_i].startswith('('):
            curr_lv += 1
        else:
            closed_lvs = int(tree_toks[tok_i].count(')'))
            if curr_lv - closed_lvs <= lv:
                tree_toks[tok_i] = ')'*(closed_lvs - (curr_lv-lv))
            curr_lv -= closed_lvs
        if lv >= curr_lv and tree_toks[tok_i]!='':
            tok_i += 1
        else:
            tree_toks.pop(tok_i)
    return ' '.join(tree_toks)


def get_lin_parse_tree(sens):
    tree_parses = const_parser.predict(sens, lang='en', verbose=False)
    lin_parses = []
    for p in tree_parses:
        lin_p = repr(p)
        if lin_p.startswith("(TOP"):
            lin_p = lin_p[5:-1]
        lin_parses.append(lin_p)
    return lin_parses


def template_match(l1, l2, lv, verbose=0, filter_empty=True):
    if filter_empty:
        not_empty1 = [any([c != " " for c in li1]) for li1 in l1]
        not_empty2 = [any([c != " " for c in li2]) for li2 in l2]
        l1 = [li1 for li1, ne1, ne2 in zip(l1, not_empty1, not_empty2) if ne1 and ne2]
        l2 = [li2 for li2, ne1, ne2 in zip(l2, not_empty1, not_empty2) if ne1 and ne2]
    docs1, docs2 = get_lin_parse_tree(l1), get_lin_parse_tree(l2)
    temps1 = [truncate_tree(doc, lv) for doc in docs1]
    temps2 = [truncate_tree(doc, lv) for doc in docs2]
    if verbose:
        for l, t in zip(l1+l2, temps1+temps2):
            print(l, "-->", t)
        print("+++++++++++++++++++++++++")
    return [int(t1 == t2) for t1, t2 in zip(temps1, temps2)]


def my_repeat(tens, n):
    return tens.unsqueeze(0).expand(n, *tens.shape).reshape(tens.shape[0]*n, *tens.shape[1:])


def l2_sim(a, b):
    dist = (a-b).square().sum(-1).sqrt()
    sim = 1/(1+dist)
    return sim
