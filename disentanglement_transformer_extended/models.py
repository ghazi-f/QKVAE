from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import SGD
import numpy as np
from tqdm import tqdm
import pandas as pd

from disentanglement_transformer_extended.h_params import *
from disentanglement_transformer_extended.graphs import get_vanilla_graph, get_vanilla_Transformer_graph
from components.links import CoattentiveTransformerLink, ConditionalCoattentiveTransformerLink
from components.bayesnets import BayesNet
from components.criteria import Supervision
from components.latent_variables import MultiCategorical
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


import matplotlib.pyplot as plt
import seaborn as sns
import itertools
sns.set_style("ticks", {"xtick.major.color": 'white', "ytick.major.color": 'white'})

import spacy_udpipe
# nlp = spacy_udpipe.load("en")
nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")

predictor = None#Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

# ============================================= DISENTANGLEMENT MODEL CLASS ============================================

class DisentanglementTransformerVAE(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, tag_index, h_params, autoload=True, wvs=None, dataset=None):
        self.dataset = dataset
        super(DisentanglementTransformerVAE, self).__init__()

        self.h_params = h_params
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

        # Setting up categorical variable indexes
        self.index = {self.generated_v: vocab_index, "sup":tag_index}


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

    def forward(self, samples, eval=False, prev_states=None, force_iw=None):
        # Just propagating values through the bayesian networks to get summaries
        if prev_states:
            infer_prev, gen_prev = prev_states
        else:
            infer_prev, gen_prev = None, None

        #                          ----------- Unsupervised Forward ----------------
        # Forward pass
        infer_inputs = {'x': samples['x'],  'x_prev': samples['x_prev']}

        # ======= Inference forward ==============
        if self.h_params.lv_kl_coeff > 0 or self.h_params.sup_coeff > 0 :
            CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = True, True
        infer_prev = self.infer_bn(infer_inputs, n_iw=self.h_params.testing_iw_samples, eval=eval,
                                   prev_states=infer_prev, force_iw=force_iw, complete=True)
        if self.h_params.lv_kl_coeff > 0 or self.h_params.sup_coeff > 0 :
            CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = False, False

        # ======= Generation forward ==============
        gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                      **{'x': samples['x'], 'x_prev': samples['x_prev']}}
        if self.iw or force_iw:
            gen_inputs = self._harmonize_input_shapes(gen_inputs, self.h_params.testing_iw_samples)

        if  self.h_params.dec_sup_coeff > 0:
            CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = True, True
        if self.step < self.h_params.anneal_kl[0]:
            gen_prev = self.gen_bn(gen_inputs, target=self.generated_v, eval=eval, prev_states=gen_prev,
                                   complete=True)
        else:
            gen_prev = self.gen_bn(gen_inputs, eval=eval, prev_states=gen_prev, complete=True)
        if self.h_params.dec_sup_coeff > 0:
            CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = False, False

        if self.h_params.lv_kl_coeff > 0 and not force_iw:
            kl_loss = self.get_lv_kl_loss()
            self.writer.add_scalar('test' + '/' + 'kl_loss', kl_loss, self.step)
        if self.h_params.sup_coeff > 0 and not force_iw:
            assert 'sup' in samples
            sup_loss = self.get_sup_att_loss(samples['sup'])
            self.writer.add_scalar('test' + '/' + 'sup_loss', sup_loss, self.step)
        if self.h_params.dec_sup_coeff > 0 and not force_iw:
            assert 'sup' in samples
            sup_loss = self.get_sup_att_loss_dec(samples['sup'])
            self.writer.add_scalar('test' + '/' + 'dec_sup_loss', sup_loss, self.step)

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

        # ======= Inference forward ==============
        if self.h_params.lv_kl_coeff > 0 or self.h_params.sup_coeff > 0 :
            CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = True, True

        if self.iw:  # and (self.step >= self.h_params.anneal_kl[0]):
            self.infer_last_states = self.infer_bn(infer_inputs, n_iw=self.h_params.training_iw_samples,
                                                   prev_states=self.infer_last_states, complete=True)
        else:
            self.infer_last_states = self.infer_bn(infer_inputs, prev_states=self.infer_last_states, complete=True)

        if self.h_params.lv_kl_coeff > 0 or self.h_params.sup_coeff > 0 :
            CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = False, False

        # ======= Generation forward ==============
        gen_inputs = {**{k.name: v for k, v in self.infer_bn.variables_hat.items()},
                      **{'x': samples['x'], 'x_prev': samples['x_prev']}}
        if self.iw:
            gen_inputs = self._harmonize_input_shapes(gen_inputs, self.h_params.training_iw_samples)

        if self.h_params.dec_sup_coeff > 0:
            ConditionalCoattentiveTransformerLink.get_att = True
        if self.step < self.h_params.anneal_kl[0]:
            self.gen_last_states = self.gen_bn(gen_inputs, target=self.generated_v,
                                               prev_states=self.gen_last_states)
        else:
            self.gen_last_states = self.gen_bn(gen_inputs, prev_states=self.gen_last_states)

        if self.h_params.dec_sup_coeff > 0:
            ConditionalCoattentiveTransformerLink.get_att = False

        # Loss computation and backward pass
        losses_uns = [loss.get_loss() * loss.w for loss in self.losses if not isinstance(loss, Supervision)]
        if self.h_params.lv_kl_coeff > 0:
            kl_loss = self.get_lv_kl_loss()
            self.writer.add_scalar('train' + '/' + 'kl_loss', kl_loss, self.step)
            losses_uns.append(self.h_params.lv_kl_coeff * kl_loss)
        if self.h_params.sup_coeff > 0:
            assert 'sup' in samples
            sup_loss = self.get_sup_att_loss(samples['sup'])
            self.writer.add_scalar('train' + '/' + 'sup_loss', sup_loss, self.step)
            losses_uns.append(self.h_params.sup_coeff * sup_loss)
        if self.h_params.dec_sup_coeff > 0:
            assert 'sup' in samples
            sup_loss = self.get_sup_att_loss_dec(samples['sup'])
            self.writer.add_scalar('train' + '/' + 'dec_sup_loss', sup_loss, self.step)
            losses_uns.append(self.h_params.dec_sup_coeff * sup_loss)
        sum(losses_uns).backward()

        if not self.h_params.contiguous_lm:
            self.infer_last_states, self.gen_last_states = None, None

        if (self.step % self.h_params.grad_accumulation_steps) == (self.h_params.grad_accumulation_steps-1):
            # Applying gradients and gradient clipping if accumulation is over
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.h_params.grad_clip)
            self.optimizer.step()
        self.step += 1

        self._dump_train_viz()
        total_loss = sum(losses_uns)

        return total_loss

    def data_specific_metrics(self):
        # this is supposed to output a list of (summary type, summary name, summary data) triplets
        with torch.no_grad():
            summary_triplets = [
                ('text', '/ground_truth', self.decode_to_text(self.gen_bn.variables_star[self.generated_v])),
                ('text', '/reconstructions', self.decode_to_text(self.generated_v.post_params['logits'])),
            ]

            n_samples = sum(self.h_params.n_latents)
            repeats = 2
            go_symbol = torch.ones([n_samples*repeats + 2 + (2 if 'zlstm' in self.gen_bn.name_to_v else 0)]).long() * \
                        self.index[self.generated_v].stoi['<go>']
            go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
            x_prev = go_symbol
            temp = 1.0
            only_z_sampling = True
            gen_len = self.h_params.max_len * (3 if self.h_params.contiguous_lm else 1)
            z_gen = self.gen_bn.name_to_v['z1']
            # When z_gen is independent from X_prev (sequence level z)
            if z_gen not in self.gen_bn.parent:
                if not (type(self.h_params.n_latents) == int and self.h_params.n_latents == 1):
                    # Getting original 2 sentences
                    orig_z_sample_1 = z_gen.prior_sample((1,))[0]
                    orig_z_sample_2 = z_gen.prior_sample((1,))[0]

                    child_zs = [self.gen_bn.name_to_v['z{}'.format(i)] for i in range(2, len(self.h_params.n_latents)+1)]
                    self.gen_bn({'z1': orig_z_sample_1.unsqueeze(1),
                                 'x_prev':torch.zeros((1, 1, self.generated_v.size)).to(self.h_params.device)})
                    orig_child_zs_1 = [z.post_samples.squeeze(1) for z in child_zs]
                    self.gen_bn({'z1': orig_z_sample_2.unsqueeze(1),
                                 'x_prev':torch.zeros((1, 1, self.generated_v.size)).to(self.h_params.device)})
                    orig_child_zs_2 = [z.post_samples.squeeze(1) for z in child_zs]
                    # Creating latent variable duplicates
                    orig_zs_samples_1 = [orig_z_sample_1] + orig_child_zs_1
                    orig_zs_samples_2 = [orig_z_sample_2] + orig_child_zs_2
                    zs_samples_1 = [orig_s.repeat(n_samples+1+(1 if 'zlstm' in self.gen_bn.name_to_v else 0), 1)
                                         for orig_s in orig_zs_samples_1]
                    zs_samples_2 = [orig_s.repeat(n_samples+1+(1 if 'zlstm' in self.gen_bn.name_to_v else 0), 1)
                                         for orig_s in orig_zs_samples_2]
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

                else:
                    z_sample = z_gen.prior_sample((n_samples, ))[0]
                    z_input = {'z1': z_sample.unsqueeze(1)}
            else:
                z_input = {}
            if ('zlstm' in self.gen_bn.name_to_v) and (self.gen_bn.name_to_v['zlstm'] not in self.gen_bn.parent):
                # case where zlstm is independant of z
                # Special case where generation is not autoregressive
                zlstm = self.gen_bn.name_to_v['zlstm']
                zlstm_sample1 = zlstm.prior_sample((1,))[0]
                zlstm_sample2 = zlstm.prior_sample((1,))[0]
                zlstm_sample = torch.cat([zlstm_sample1.repeat(n_samples+1, 1), zlstm_sample2,
                                          zlstm_sample2.repeat(n_samples+1, 1), zlstm_sample1], 0)
                self.gen_bn({'z1': z_sample.unsqueeze(1).expand(z_sample.shape[0], gen_len, z_sample.shape[1]),
                             'zlstm': zlstm_sample.unsqueeze(1).expand(z_sample.shape[0], gen_len, z_sample.shape[1])})
                samples_i = self.generated_v.post_params['logits']
                x_prev = torch.argmax(samples_i, dim=-1)

            else:
                # Normal Autoregressive generation
                for i in range(gen_len):
                    self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i+1, v.shape[-1])
                                                      for k, v in z_input.items()}})
                    if only_z_sampling:
                        samples_i = self.generated_v.post_params['logits']
                    else:
                        samples_i = self.generated_v.posterior(logits=self.generated_v.post_params['logits'],
                                                               temperature=temp).rsample()
                    x_prev = torch.cat([x_prev, torch.argmax(samples_i,     dim=-1)[..., -1].unsqueeze(-1)],
                                       dim=-1)

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
            text = ' |||| '.join([' '.join([self.index[self.generated_v].itos[w]
                                            for w in sen]).replace('!', '<eos>').replace('.', '<eos>')
                                 .replace('?', '<eos>')#.split('<eos>')[0]
                                  for sen in x_hat_params]).replace('<pad>', '_').replace('_unk', '<?>').replace('<eos>', '\n')
        else:
            samples = [' '.join([self.index[self.generated_v].itos[w]
                                 for w in sen]).replace('!', '<eos>').replace('.', '<eos>').replace('?', '<eos>')
                           .split('<eos>')[0].replace('<go>', '').replace('</go>', '')
                       for sen in x_hat_params]
            if self.dataset == 'yelp':
                samples = [sen.split('<eos>')[0] for sen in samples]
            first_sample, second_sample = samples[:int(len(samples)/2)], samples[int(len(samples) / 2):]
            samples = ['**First Sample**\n'] + \
                      [('orig' if i == 0 else str(i-1) if sample == first_sample[0] else '**'+str(i-1)+'**') + ': ' +
                       sample for i, sample in enumerate(first_sample)] + \
                      ['**Second Sample**\n'] + \
                      [('orig' if i == 0 else str(i-1) if sample == second_sample[0] else '**'+str(i-1)+'**') + ': ' +
                       sample for i, sample in enumerate(second_sample)]
            text = ' |||| '.join(samples).replace('<pad>', '_').replace('_unk', '<?>')

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

        samples = [' '.join([vocab_index.itos[w]
                             for w in sen]).replace('!', '<eos>').replace('.', '<eos>').replace('?', '<eos>')
                       .split('<eos>')[0].replace('<go>', '').replace('</go>', '')
                       .replace('<pad>', '_').replace('_unk', '<?>')
                   for sen in x_hat_params]

        return samples

    def get_perplexity(self, iterator):
        with torch.no_grad():
            neg_log_perplexity_lb = 0
            total_samples = 0
            infer_prev, gen_prev = None, None
            force_iw = ['z{}'.format(len(self.h_params.n_latents))]
            iwlbo = IWLBo(self, 1)

            for i, batch in enumerate(tqdm(iterator, desc="Getting Model Perplexity")):
                if batch.text.shape[1] < 2: continue
                inp = {'x': batch.text[..., 1:], 'x_prev': batch.text[..., :-1]}
                if self.h_params.sup_coeff > 0:
                    inp['sup'] = batch.label
                infer_prev, gen_prev = self(inp, prev_states=(infer_prev, gen_prev), force_iw=force_iw)
                if not self.h_params.contiguous_lm:
                    infer_prev, gen_prev = None, None
                elbo = - iwlbo.get_loss(actual=True)
                batch_size = batch.text.shape[0]
                total_samples_i = torch.sum(batch.text != self.h_params.vocab_ignore_index)
                neg_log_perplexity_lb += elbo * batch_size

                total_samples += total_samples_i

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
                inp = {'x': batch.text[..., 1:], 'x_prev': batch.text[..., :-1]}
                if self.h_params.sup_coeff > 0:
                    inp['sup'] = batch.label
                infer_prev, gen_prev = self(inp, prev_states=(infer_prev, gen_prev))
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
                    self.index[self.generated_v].stoi['<go>']
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
        self.gen_bn({**{'z{}'.format(i+1): orig_zs[i].unsqueeze(1) for i in range(len(orig_zs))},
                     'x_prev': torch.zeros((n_samples * n_orig_sentences, 1, self.generated_v.size)).to(
                         self.h_params.device)})
        zs_sample = [zs[0].prior_sample((n_samples * n_orig_sentences,))[0]] +\
                    [z.post_samples.squeeze(1) for z in zs[1:]]

        for id in var_z_ids:
            assert id < sum(h_params.n_latents)
            z_number = sum([id > sum(h_params.n_latents[:i + 1]) for i in range(len(h_params.n_latents))])
            z_index = id - sum(h_params.n_latents[:z_number])
            start, end = int(h_params.z_size / max(h_params.n_latents) * z_index), int(
                h_params.z_size / max(h_params.n_latents) * (z_index + 1))
            source, destination = zs_sample[z_number], orig_zs[z_number]
            destination[:, start:end] = source[:, start:end]

        z_input = {'z{}'.format(i+1): orig_zs[i].unsqueeze(1) for i in range(len(orig_zs))}

        # Normal Autoregressive generation
        for i in range(gen_len):
            self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i + 1, v.shape[-1])
                                             for k, v in z_input.items()}})
            samples_i = self.generated_v.post_params['logits']

            x_prev = torch.cat([x_prev, torch.argmax(samples_i, dim=-1)[..., -1].unsqueeze(-1)],
                               dim=-1)

        text = self.decode_to_text2(x_prev, self.h_params.vocab_size, self.index[self.generated_v])
        return text, {'z{}'.format(i+1): zs_sample[i].tolist() for i in range(len(orig_zs))}

    def get_sentences(self, n_samples, gen_len=16, sample_w=False, vary_z=True, complete=None, contains=None,
                      max_tries=100):
        n_latents = self.h_params.n_latents
        final_text, final_samples, final_params = [], {'z{}'.format(i+1): [] for i in range(len(n_latents))},\
                                                  {'z{}'.format(i+1): None for i in range(1, len(n_latents))}
        trys = 0
        while n_samples > 0:
            trys += 1
            go_symbol = torch.ones([n_samples]).long() * \
                        self.index[self.generated_v].stoi['<go>']
            go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
            x_prev = go_symbol
            if complete is not None:
                for token in complete.split(' '):
                    x_prev = torch.cat([x_prev, torch.ones([n_samples, 1]).long().to(self.h_params.device) * \
                                        self.index[self.generated_v].stoi[token]], dim=1)
                gen_len = gen_len - len(complete.split(' '))
            temp = 1.
            z_gen = self.gen_bn.name_to_v['z1']
            if vary_z:
                z_sample = z_gen.prior_sample((n_samples,))[0]
            else:
                z_sample = z_gen.prior_sample((1,))[0]
                z_sample = z_sample.repeat(n_samples, 1)
            child_zs = [self.gen_bn.name_to_v['z{}'.format(i)] for i in range(2, len(self.h_params.n_latents) + 1)]

            # Structured Z case
            if vary_z:
                self.gen_bn({'z1': z_sample.unsqueeze(1),
                            'x_prev': torch.zeros((n_samples, 1, self.generated_v.size)).to(self.h_params.device)})
                zs_samples = [z_sample] + [z.post_samples.squeeze(1) for z in child_zs]
                zs_params = {z.name: z.post_params for z in child_zs}
            else:
                self.gen_bn({'z1': z_sample[0].unsqueeze(0).unsqueeze(1),
                            'x_prev': torch.zeros((1, 1, self.generated_v.size)).to(self.h_params.device)})
                zs_samples = [z_sample] + [z.post_samples.squeeze(1).repeat(n_samples, 1) for z in child_zs]
                zs_params = {z.name: {k: v.squeeze(1).repeat(n_samples, 1) for k, v in z.post_params.items()}
                             for z in child_zs}

            z_input = {'z{}'.format(i+1): z_s.unsqueeze(1) for i, z_s in enumerate(zs_samples)}
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
                return text, {'z{}'.format(i+1): z_s for i, z_s in enumerate(zs_samples)}, zs_params
            else:
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
        if self.dataset == 'de_nli':
            shall_dep_func = lambda x: shallow_dependencies2(x, de_nlp, ['sb', 'verb', 'oa', 'da', 'op', 'oc'])
        else:
            shall_dep_func = lambda x: shallow_dependencies2(x, nlp)
        text_sents = [' '.join([self.index[self.generated_v].itos[w]
                                for w in s]).replace(' <pad>', '').replace(' <eos>', '')
                      for s in text_in]
        # Getting relations' positions
        rel_idx = [out['idx'] for out in shall_dep_func(text_sents)]
        # Getting layer wise attention values

        CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = True, True
        self.infer_bn({'x': text_in})
        att_vals = self.get_enc_att_vals().cpu().detach().numpy()
        att_vals = att_vals.mean(-2)
        CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = False, False
        att_maxes = att_vals.argmax(-1).tolist()
        return rel_idx, att_maxes

    def get_enc_att_vals(self):
        max_len = self.h_params.max_len-1
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
                if lv_layer == 2:  # TODO: update this part for structured inference networks
                    att_out[..., -1] *= 0
                var_att_weights.append(att_out)
                # var_att_weights.append(att_out.cpu().detach().numpy())
            # att_weights.append(var_att_weights)
            att_weights.append(torch.stack(var_att_weights))
        # att_vals shape:[sent, lv, layer, tok]
        att_vals = torch.stack(att_weights).transpose(2, 1).transpose(1, 0)
        # att_vals = np.transpose(np.array(att_weights), (2, 0, 1, 3))
        return att_vals

    def get_lv_kl_loss(self):
        att_vals = self.get_enc_att_vals()
        kl_list = []
        n_lv = sum(self.h_params.n_latents)
        # Calculating pairwise KL divergences between latent variable attention distributions
        for i in range(n_lv):
            for j in range(n_lv):
                if i!=j:
                    logit0, logit1 = att_vals[:, i], att_vals[:, j]
                    kl_per_dim = torch.softmax(logit0, dim=-1)*(torch.log_softmax(logit0, dim=-1) -
                                                                torch.log_softmax(logit1, dim=-1))
                    kl_list.append(torch.sum(kl_per_dim, dim=-1).unsqueeze(0))
        return - torch.mean(torch.cat(kl_list))

    def get_sup_att_loss(self, sup):
        if self.h_params.sup_loss_choice == 'multi':
            return self._get_sup_att_loss_multi(sup)
        elif self.h_params.sup_loss_choice == 'single':
            return self._get_sup_att_loss_single(sup)
        else:
            raise NotImplementedError("Unrecognized supervised loss "
                                      "choice :\"{}\"".format(self.h_params.sup_loss_choice))

    def _get_sup_att_loss_single(self, sup):
        # This performs mutli_class classification with attention values
        # att_vals shape:[sent, lv, layer, tok+1] the +1 is for the rest of the attention outside of the sentence
        att_vals = self.get_enc_att_vals()[..., :-1]
        # averaging over layers which leads to shape [sent, lv, tok]
        att_vals = att_vals.mean(-2)
        # Avoiding the importance sampled forward passes
        if len(att_vals.shape)>3:
            return 0.

        # On_hot representing sup, removing "other syntactic roles" label and calculating mask
        o_idx = self.index['sup'].stoi['o']
        unk_idx = self.index['sup'].unk_index
        sup = F.one_hot(sup, len(self.index['sup']))
        role_idx = list(range(sup.shape[-1]))
        role_idx.remove(unk_idx)
        role_idx.remove(o_idx)
        sup = sup[..., role_idx].transpose(-1, -2)

        #Making new dimension for classification
        att_vals = torch.cat([att_vals.unsqueeze(-1), 1-att_vals.unsqueeze(-1)], -1)
        sup = torch.cat([sup.unsqueeze(-1), 1-sup.unsqueeze(-1)], -1)


        # Calculating cross entropy
        c_e = -(torch.log(att_vals)*sup).mean()
        return c_e

    def _get_sup_att_loss_multi(self, sup):
        # For this one, each target LV is a classifier for a determined syntactic role
        # att_vals shape:[sent, lv, layer, tok+1] the +1 is for the rest of the attention outside of the sentence
        att_vals = self.get_enc_att_vals()[..., :-1]
        # averaging over layers which leads to shape [sent, lv, tok]
        att_vals = att_vals.mean(-2)
        # Avoiding the importance sampled forward passes
        if len(att_vals.shape)>3:
            return 0.
        # transposing lv and tok dimension to get the softmax right
        att_vals = att_vals.transpose(-1, -2)
        # On_hot representing sup, removing "other syntactic roles" label and calculating mask
        o_idx = self.index['sup'].stoi['o']
        unk_idx = self.index['sup'].unk_index
        mask = 1-(sup == o_idx).int().unsqueeze(-1)
        sup = F.one_hot(sup, len(self.index['sup']))
        role_idx = list(range(sup.shape[-1]))
        role_idx.remove(unk_idx)
        role_idx.remove(o_idx)
        sup = sup[..., role_idx]
        # Making it so that values sum to 1 on the lv axis without a softmax
        att_vals = att_vals / att_vals.sum(-1).unsqueeze(-1)

        # Calculating cross entropy
        c_e = -(torch.log(att_vals)*sup*mask).sum()/mask.sum()
        return c_e

    def get_dec_att(self):
        # att_weights shape:[layer, sent, tok, lv] -->[sent, layer, tok, lv]
        att_weights = self.gen_bn.approximator[self.generated_v].att_vals
        att_vals = torch.stack(att_weights).transpose(1, 0)
        return att_vals

    def get_sup_att_loss_dec(self, sup):
        if self.h_params.sup_loss_choice == 'multi':
            return self._get_sup_att_loss_dec_multi(sup)
        elif self.h_params.sup_loss_choice == 'single':
            return self._get_sup_att_loss_dec_single(sup)
        else:
            raise NotImplementedError("Unrecognized supervised decoder loss "
                                      "choice :\"{}\"".format(self.h_params.sup_loss_choice))

    def _get_sup_att_loss_dec_multi(self, sup):
        # att_vals shape:[sent, layer, tok, lv], averaging over layers which leads to shape [sent, tok, lv]
        att_vals = self.get_dec_att().mean(1)
        # Avoiding the importance sampled forward passes
        if len(att_vals.shape)>3:
            return 0.
        # One_hot representing sup, removing "other syntactic roles" label and calculating mask
        o_idx = self.index['sup'].stoi['o']
        unk_idx = self.index['sup'].unk_index
        mask = 1-(sup == o_idx).int().unsqueeze(-1)
        sup = F.one_hot(sup, len(self.index['sup']))
        role_idx = list(range(sup.shape[-1]))
        role_idx.remove(unk_idx)
        role_idx.remove(o_idx)
        sup = sup[..., role_idx]

        # Calculating cross entropy
        c_e = -(torch.log(att_vals)*sup*mask).sum()/mask.sum()
        return c_e

    def _get_sup_att_loss_dec_single(self, sup):
        # att_vals shape:[sent, layer, tok, lv], average on layers and transpose which leads to shape [sent, lv, tok]
        att_vals = self.get_dec_att().mean(1).transpose(-1, -2)
        # Avoiding the importance sampled forward passes
        if len(att_vals.shape)>3:
            return 0.
        # On_hot representing sup, removing "other syntactic roles" label and calculating mask
        o_idx = self.index['sup'].stoi['o']
        unk_idx = self.index['sup'].unk_index
        sup = F.one_hot(sup, len(self.index['sup']))
        role_idx = list(range(sup.shape[-1]))
        role_idx.remove(unk_idx)
        role_idx.remove(o_idx)
        sup = sup[..., role_idx].transpose(-1, -2)

        #Making new dimension for classification
        att_vals = torch.cat([att_vals.unsqueeze(-1), 1-att_vals.unsqueeze(-1)], -1)
        sup = torch.cat([sup.unsqueeze(-1), 1-sup.unsqueeze(-1)], -1)


        # Calculating cross entropy
        c_e = -(torch.log(att_vals)*sup).mean()
        return c_e

    def get_att_and_rel_idx_all(self, text_in, roles=None):
        roles = roles if roles is not None else['nsubj', 'verb', 'dobj', 'pobj']
        max_len = text_in.shape[-1]
        text_sents = [' '.join([self.index[self.generated_v].itos[w]
                                for w in s]).replace(' <pad>', '').replace(' <eos>', '')
                      for s in text_in]
        # Getting relations' positions
        rel_idx = [out['idx'] for out in shallow_dependencies2(text_sents, nlp, roles)]
        # Getting layer wise attention values

        CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = True, True
        self.infer_bn({'x': text_in})
        CoattentiveTransformerLink.get_att, ConditionalCoattentiveTransformerLink.get_att = False, False
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
                if lv_layer == 2:  # TODO: update this part for structured inference networks
                    att_out[..., -1] *= 0
                var_att_weights.append(att_out.cpu().detach().numpy())
            att_weights.append(var_att_weights)
        # att_vals shape:[sent, lv, layer, tok]
        att_vals = np.transpose(np.array(att_weights), (2, 0, 1, 3)).mean(-2)
        att_maxes = att_vals.argmax(-1).tolist()
        return rel_idx, att_maxes

    def get_encoder_disentanglement_score(self, data_iter):
        rel_idx, att_maxes = [], []
        for i, batch in enumerate(tqdm(data_iter, desc="Getting model relationship accuracy")):
            rel_idx_i, att_maxes_i = self.get_att_and_rel_idx(batch.text[..., 1:])
            rel_idx.extend(rel_idx_i)
            att_maxes.extend(att_maxes_i)

        lv_scores = []
        for lv in range(sum(self.h_params.n_latents)):
            found = {'nsubj': [], 'verb': [], 'dobj': [], 'pobj': []} if self.dataset != "de_nli" else \
                {'sb': [], 'verb': [], 'oa': [], 'da': [], 'op': [], 'oc': []}
            for att, rel_pos in zip(att_maxes, rel_idx):
                for k in found.keys():
                    if len(rel_pos[k]):
                        found[k].append(att[lv] in rel_pos[k])
            lv_scores.append(found)
        enc_att_scores = {'nsubj': [], 'verb': [], 'dobj': [], 'pobj': []}if self.dataset != "de_nli" else \
                {'sb': [], 'verb': [], 'oa': [], 'da': [], 'op': [], 'oc': []}
        for lv in range(sum(self.h_params.n_latents)):
            for k, v in lv_scores[lv].items():
                enc_att_scores[k].append(np.mean(v))

        enc_max_score, enc_disent_score, enc_disent_vars = {}, {}, {}
        for k, v in enc_att_scores.items():
            sort_idx = np.argsort(v)
            enc_disent_vars[k], enc_disent_score[k], enc_max_score[k] = \
                sort_idx[-1], v[sort_idx[-1]] - v[sort_idx[-2]], v[sort_idx[-1]]
        return enc_att_scores, enc_max_score, enc_disent_score, enc_disent_vars

    def get_encoder_disentanglement_score_all(self, data_iter, roles=None):
        roles = roles if roles is not None else['nsubj', 'verb', 'dobj', 'pobj']
        rel_idx, att_maxes = [], []
        for i, batch in enumerate(tqdm(data_iter, desc="Getting model relationship accuracy")):
            rel_idx_i, att_maxes_i = self.get_att_and_rel_idx_all(batch.text[..., 1:], roles)
            rel_idx.extend(rel_idx_i)
            att_maxes.extend(att_maxes_i)

        lv_scores = []
        for lv in range(sum(self.h_params.n_latents)):
            found = {r: [] for r in roles}
            for att, rel_pos in zip(att_maxes, rel_idx):
                for k in found.keys():
                    if len(rel_pos[k]):
                        found[k].append(att[lv] in rel_pos[k])
            lv_scores.append(found)
        enc_att_scores = {r:[] for r in roles}
        for lv in range(sum(self.h_params.n_latents)):
            for k, v in lv_scores[lv].items():
                enc_att_scores[k].append(np.mean(v))

        enc_max_score, enc_disent_score, enc_disent_vars = {}, {}, {}
        for k, v in enc_att_scores.items():
            sort_idx = np.argsort(v)
            enc_disent_vars[k], enc_disent_score[k], enc_max_score[k] = \
                sort_idx[-1], v[sort_idx[-1]] - v[sort_idx[-2]], v[sort_idx[-1]]
        return enc_att_scores, enc_max_score, enc_disent_score, enc_disent_vars

    def get_sentence_statistics2(self, orig, sen, orig_relations, alt_relations, orig_temp, alt_temp):
        orig_relations, alt_relations = orig_relations['text'], alt_relations['text']
        same_struct = True
        for k in orig_relations.keys():
            if (orig_relations[k] == '' and alt_relations[k] != '') or (orig_relations[k] == '' and alt_relations[k] != ''):
                same_struct = False

        def get_diff(arg):
            if orig_relations[arg] == '' and alt_relations[arg] == '':
                return False, False, False
            elif orig_relations[arg] != '' and alt_relations[arg] != '':
                return orig_relations[arg] != alt_relations[arg], False, True
            else:
                return False, orig_relations[arg] != alt_relations[arg], False
        syn_temp_diff = orig_temp['syn'] != alt_temp['syn']
        lex_temp_diff = orig_temp['lex'] != alt_temp['lex']

        rels = ['nsubj', 'verb', 'dobj', 'pobj'] if self.dataset != "de_nli" else ['sb', 'verb', 'oa', 'da', 'op', 'oc']
        return [get_diff(r) for r in rels], same_struct, \
                   syn_temp_diff, lex_temp_diff

    def get_sem_stats(self, orig_relations, alt_relations):
        orig_relations, alt_relations = orig_relations['text'], alt_relations['text']

        def get_diff(arg):
            if orig_relations[arg] == '' and alt_relations[arg] == '':
                return False, False, False
            elif orig_relations[arg] != '' and alt_relations[arg] != '':
                return orig_relations[arg] != alt_relations[arg], False, True
            else:
                return False, orig_relations[arg] != alt_relations[arg], False

        rels = ['V', 'ARG0', 'ARG1']
        return [get_diff(r) for r in rels]

    def get_sentence_statistics_all(self, orig, sen, orig_relations, alt_relations, orig_temp, alt_temp, roles):
        roles = roles if roles is not None else['nsubj', 'verb', 'dobj', 'pobj']
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
        diffs = [get_diff(r) for r in roles]
        return diffs, same_struct, \
               syn_temp_diff, lex_temp_diff

    def swap_latent(self, src_lvs, trg_lvs, var_z_ids, gen_len, complete=None, no_unk=True):
        h_params = self.h_params
        has_struct = 'zs' in self.gen_bn.name_to_v
        assert src_lvs['z1'].shape[0] == trg_lvs['z1'].shape[0]
        n_orig_sentences = trg_lvs['z1'].shape[0]
        n_samples = 1
        go_symbol = torch.ones([n_samples * n_orig_sentences]).long() * \
                    self.index[self.generated_v].stoi['<go>']
        go_symbol = go_symbol.to(self.h_params.device).unsqueeze(-1)
        x_prev = go_symbol
        if complete is not None:
            for token in complete.split(' '):
                x_prev = torch.cat([x_prev, torch.ones([n_samples * n_orig_sentences, 1]).long().to(self.h_params.device) * \
                    self.index[self.generated_v].stoi[token]], dim=1)
            gen_len = gen_len - len(complete.split(' '))

        orig_z = trg_lvs['z1'].unsqueeze(1)
        z_sample = src_lvs['z1'].reshape(n_samples*n_orig_sentences, -1)
        orig_z = orig_z.transpose(0, 1).reshape(n_samples*n_orig_sentences, -1)

        for id in var_z_ids:
            if id < sum(h_params.n_latents):
                z_number = sum([id> sum(h_params.n_latents[:i+1]) for i in range(len(h_params.n_latents))])
                z_index = id - sum(h_params.n_latents[:z_number])
                start, end = int(h_params.z_size/max(h_params.n_latents)*z_index),\
                             int(h_params.z_size/max(h_params.n_latents)*(z_index+1))
                source, destination = [z_sample][z_number], [orig_z][z_number]
                destination[:, start:end] = source[:, start:end]
            elif id == sum(h_params.n_latents) and has_struct:
                raise NotImplementedError("Didn't Implement this for structure embeddings yet.")
            else:
                raise IndexError("You gave a too high z_id for swapping with this model")

        z_input = {'z1': orig_z.unsqueeze(1)}

        # Normal Autoregressive generation
        for i in range(gen_len):
            self.gen_bn({'x_prev': x_prev, **{k: v.expand(v.shape[0], i+1, v.shape[-1])
                                              for k, v in z_input.items()}})
            samples_i = self.generated_v.post_params['logits']
            if no_unk:
                annul_vector = 1-F.one_hot(torch.tensor([self.index[self.generated_v].stoi['<unk>']]).to(self.h_params.device),
                                           self.h_params.vocab_size)
                samples_i *= annul_vector

            x_prev = torch.cat([x_prev, torch.argmax(samples_i,     dim=-1)[..., -1].unsqueeze(-1)],
                               dim=-1)

        text = self.decode_to_text2(x_prev, self.h_params.vocab_size, self.index[self.generated_v])
        return text, {'z1': orig_z}

    def _get_stat_data_frame_recon(self, data_iter, batch_size=400, delta_drop=True):
        # Getting the right parsing function
        stats = []
        rels = ['nsubj', 'verb', 'dobj', 'pobj'] if self.dataset != "de_nli" else ['sb', 'verb', 'oa', 'da', 'op', 'oc']
        sems = ['V', 'ARG0', 'ARG1']
        if self.dataset == 'de_nli':
            shall_dep_func = lambda x: shallow_dependencies2(x, de_nlp, ['sb', 'verb', 'oa', 'da', 'op', 'oc'])
        else:
            shall_dep_func = lambda x: shallow_dependencies2(x, nlp)
        z_infer, x_gen = {'z{}'.format(i + 1): self.infer_bn.name_to_v['z{}'.format(i + 1)]
                          for i in range(len(self.h_params.n_latents))}, self.gen_bn.name_to_v['x']
        # Collecting original sentences with their parses and lv samples
        text, samples, orig_rels, orig_sem, orig_temps = [], {"z1": []}, [], [], []
        batch_collec = []
        for i, batch_i in enumerate(tqdm(data_iter, desc="Encoding and reconstructing sentences")):
            batch_collec.append(batch_i)
            if sum([b.text.shape[0] for b in batch_collec])>= batch_size:
                batch_text = torch.cat([b.text for b in batch_collec], dim=0)
                batch_collec = []
                if batch_text.shape[1] < 2: continue
                bs = batch_text.shape[0]
                self.infer_bn({'x': batch_text[..., 1:]})
                samples["z1"].extend(torch.cat([v.post_params['loc'][..., 0, :] for k, v in z_infer.items()], dim=-1).tolist())
                text.extend(self._get_alternative_sentences(
                    prev_latent_vals={"z1": torch.Tensor(samples["z1"][-bs:]).to(self.h_params.device)}, params=None,
                    n_samples=1, var_z_ids=[], gen_len=self.h_params.max_len - 1)[0])
                orig_rels.extend(shall_dep_func(text[-bs:]))
                orig_sem.extend(srl_from_sents(text[-bs:]))
                orig_temps.extend(truncated_template(text[-bs:]))
        samples = {k: torch.Tensor(v).to(self.h_params.device) for k, v in samples.items()}
        n_samples = len(text)
        new_idx = np.random.choice(list(range(n_samples)), n_samples).tolist()
        for i in range(int(n_samples / batch_size)):
            for j in tqdm(range(sum(self.h_params.n_latents)), desc="Processing sample {}".format(str(i))):
                # Altering the sentences
                alt_text, _ = self.swap_latent({k: v[new_idx[i * batch_size:(i + 1) * batch_size]] for k, v in
                                             samples.items()}, {k: v[i * batch_size:(i + 1) * batch_size]
                                                                for k, v in samples.items()}, [j], 16)
                alt_rels = shall_dep_func(alt_text)
                alt_sem = srl_from_sents(alt_text)
                alt_temps = truncated_template(alt_text)
                # Getting alteration statistics
                for k in range(batch_size):
                    orig_text = text[(i * batch_size) + k % batch_size]
                    try:
                        r_diffs, same_struct, syn_temp_diff, lex_temp_diff = \
                            self.get_sentence_statistics2(orig_text, alt_text[k],
                                                     orig_rels[(i * batch_size) + k % batch_size], alt_rels[k],
                                                     orig_temps[(i * batch_size) + k % batch_size], alt_temps[k])
                        sem_diffs = self.get_sem_stats(orig_sem[(i * batch_size) + k % batch_size], alt_sem[k])
                    except RecursionError or IndexError:
                        continue
                    stats.append([orig_text, alt_text[k], j, *(int(diff[0]) for diff in r_diffs),
                                  *(int(diff[1]) for diff in r_diffs),
                                  *(int(diff[2]) for diff in r_diffs), *(int(diff[0]) for diff in sem_diffs),
                                  *(int(diff[2]) for diff in sem_diffs),same_struct, syn_temp_diff, lex_temp_diff])

        header = ['original', 'altered', 'alteration_id', *(r + "_diff" for r in rels), *(r + "_struct" for r in rels),
                  *(r + "_ex" for r in rels), *(r + "_diff" for r in sems), *(r + "_ex" for r in sems),
                  'same_struct', 'syntemp_diff', 'lextemp_diff']
        df = pd.DataFrame(stats, columns=header)
        # Getting syntactic role scores
        if not delta_drop:
            for diff_lab, struct_lab in zip([r + "_diff" for r in rels], [r + "_struct" for r in rels]):
                df[diff_lab] = [d or s for d, s in zip(df[diff_lab], df[struct_lab])]
        var_wise_scores = df.groupby('alteration_id').mean()[
            [r + "_diff" for r in rels] + ['syntemp_diff', 'lextemp_diff']]
        var_wise_scores_struct = df.groupby('alteration_id').mean()[[r + "_struct" for r in rels]]
        var_wise_scores_ex = df.groupby('alteration_id').mean()[[r + "_ex" for r in rels]]
        var_wise_scores.set_axis([a.split('_')[0] for a in var_wise_scores.axes[1]], axis=1, inplace=True)
        if delta_drop:
            # renormalizing
            ex_array = np.array(var_wise_scores_ex)+1e-10
            ex_array = np.concatenate([ex_array, np.ones((sum(self.h_params.n_latents), 2))], axis=1)
            var_wise_scores = var_wise_scores / ex_array

        disent_score, lab_wise_disent, dec_disent_vars = 0, {}, {}
        for lab in rels:
            try:
                dec_disent_vars[lab] = var_wise_scores.idxmax()[lab]
            except TypeError:
                dec_disent_vars[lab] = -1
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

        # Getting Semantic role scores (default delta drop)
        sem_var_wise_scores = df.groupby('alteration_id').mean()[
            [r + "_diff" for r in sems]]
        sem_var_wise_scores_ex = df.groupby('alteration_id').mean()[[r + "_ex" for r in sems]]
        sem_var_wise_scores.set_axis([a.split('_')[0] for a in sem_var_wise_scores.axes[1]], axis=1, inplace=True)
        if delta_drop:
            # renormalizing
            ex_array = np.array(sem_var_wise_scores_ex)+1e-10
            sem_var_wise_scores = sem_var_wise_scores / ex_array

        sem_disent_score, sem_lab_wise_disent, sem_dec_disent_vars = 0, {}, {}
        for lab in sems:
            try:
                sem_dec_disent_vars[lab] = sem_var_wise_scores.idxmax()[lab]
            except TypeError:
                sem_dec_disent_vars[lab] = -1
                sem_lab_wise_disent[lab] = 0
                continue
            top2 = np.array(sem_var_wise_scores.nlargest(2, lab)[lab])
            diff = top2[0] - top2[1]
            sem_lab_wise_disent[lab] = diff
            sem_disent_score += diff

        return disent_score, lab_wise_disent, var_wise_scores, dec_disent_vars, \
               sem_disent_score, sem_lab_wise_disent, sem_var_wise_scores, sem_dec_disent_vars

    def _get_stat_data_frame2(self, n_samples=2000, n_alterations=1, batch_size=100, delta_drop=True):
        stats = []
        rels = ['nsubj', 'verb', 'dobj', 'pobj'] if self.dataset != "de_nli" else ['sb', 'verb', 'oa', 'da', 'op', 'oc']
        if self.dataset == 'de_nli':
            shall_dep_func = lambda x: shallow_dependencies2(x, de_nlp, ['sb', 'verb', 'oa', 'da', 'op', 'oc'])
        else:
            shall_dep_func = lambda x: shallow_dependencies2(x, nlp)
        # Generating n_samples sentences
        text, samples, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                              sample_w=False, vary_z=True, complete=None)
        orig_rels = shall_dep_func(text)
        orig_temps = truncated_template(text)
        for _ in tqdm(range(int(n_samples / batch_size)), desc="Generating original sentences"):
            text_i, samples_i, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                                      sample_w=False, vary_z=True, complete=None)
            text.extend(text_i)
            for k in samples.keys():
                samples[k] = torch.cat([samples[k], samples_i[k]])
            orig_rels.extend(shall_dep_func(text_i))
            orig_temps.extend(truncated_template(text_i))
        for i in range(int(n_samples / batch_size)):
            for j in tqdm(range(sum(self.h_params.n_latents)), desc="Processing sample {}".format(str(i))):
                # Altering the sentences
                alt_text, _ = self._get_alternative_sentences(
                    prev_latent_vals={k: v[i * batch_size:(i + 1) * batch_size]
                                      for k, v in samples.items()},
                    params=None, var_z_ids=[j], n_samples=n_alterations,
                    gen_len=self.h_params.max_len - 1, complete=None)
                alt_rels = shall_dep_func(alt_text)
                alt_temps = truncated_template(alt_text)
                # Getting alteration statistics
                for k in range(n_alterations * batch_size):
                    orig_text = text[(i * batch_size) + k % batch_size]
                    try:
                        r_diffs, same_struct, syn_temp_diff, lex_temp_diff = \
                            self.get_sentence_statistics2(orig_text, alt_text[k],
                                                          orig_rels[(i * batch_size) + k % batch_size], alt_rels[k],
                                                          orig_temps[(i * batch_size) + k % batch_size], alt_temps[k])
                    except RecursionError or IndexError:
                        continue
                    stats.append([orig_text, alt_text[k], j, *(int(diff[0]) for diff in r_diffs),
                                  *(int(diff[1]) for diff in r_diffs),
                                  *(int(diff[2]) for diff in r_diffs),same_struct, syn_temp_diff, lex_temp_diff])

        header = ['original', 'altered', 'alteration_id', *(r + "_diff" for r in rels), *(r + "_struct" for r in rels),
                  *(r + "_ex" for r in rels),
                  'same_struct', 'syntemp_diff', 'lextemp_diff']
        df = pd.DataFrame(stats, columns=header)
        if not delta_drop:
            for diff_lab, struct_lab in zip([r+"_diff" for r in rels], [r+"_struct" for r in rels]):
                df[diff_lab] = [d or s for d, s in zip(df[diff_lab], df[struct_lab])]
        var_wise_scores = df.groupby('alteration_id').mean()[[r+"_diff" for r in rels]+['syntemp_diff', 'lextemp_diff']]
        var_wise_scores_struct = df.groupby('alteration_id').mean()[[r+"_struct" for r in rels]]
        var_wise_scores_ex = df.groupby('alteration_id').mean()[[r+"_ex" for r in rels]]
        var_wise_scores.set_axis([a.split('_')[0] for a in var_wise_scores.axes[1]], axis=1, inplace=True)
        if delta_drop:
            # renormalizing
            ex_array = np.array(var_wise_scores_ex)+1e-10
            ex_array = np.concatenate([ex_array, np.ones((sum(self.h_params.n_latents), 2))], axis=1)
            var_wise_scores = var_wise_scores/ex_array

        disent_score = 0
        lab_wise_disent = {}
        dec_disent_vars = {}
        for lab in rels:
            try:
                dec_disent_vars[lab] = var_wise_scores.idxmax()[lab]
            except TypeError:
                dec_disent_vars[lab] = -1
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

    def _get_stat_data_frame3(self, n_samples=2000, n_alterations=1, batch_size=100, delta_drop=True):
        stats = []
        rels = ['nsubj', 'verb', 'dobj', 'pobj'] if self.dataset != "de_nli" else ['sb', 'verb', 'oa', 'da', 'op', 'oc']
        sems = ['V', 'ARG0', 'ARG1']
        if self.dataset == 'de_nli':
            shall_dep_func = lambda x: shallow_dependencies2(x, de_nlp, ['sb', 'verb', 'oa', 'da', 'op', 'oc'])
        else:
            shall_dep_func = lambda x: shallow_dependencies2(x, nlp)
        # Generating n_samples sentences
        text, samples, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                              sample_w=False, vary_z=True, complete=None)
        orig_rels = shall_dep_func(text)
        orig_sem = srl_from_sents(text)
        orig_temps = truncated_template(text)
        for _ in tqdm(range(int(n_samples / batch_size)), desc="Generating original sentences"):
            text_i, samples_i, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                                      sample_w=False, vary_z=True, complete=None)
            text.extend(text_i)
            for k in samples.keys():
                samples[k] = torch.cat([samples[k], samples_i[k]])
            orig_rels.extend(shall_dep_func(text_i))
            orig_sem.extend(srl_from_sents(text_i))
            orig_temps.extend(truncated_template(text_i))
        for i in range(int(n_samples / batch_size)):
            for j in tqdm(range(sum(self.h_params.n_latents)), desc="Processing sample {}".format(str(i))):
                # Altering the sentences
                alt_text, _ = self._get_alternative_sentences(
                    prev_latent_vals={k: v[i * batch_size:(i + 1) * batch_size]
                                      for k, v in samples.items()},
                    params=None, var_z_ids=[j], n_samples=n_alterations,
                    gen_len=self.h_params.max_len - 1, complete=None)
                alt_rels = shall_dep_func(alt_text)
                alt_sem = srl_from_sents(alt_text)
                alt_temps = truncated_template(alt_text)
                # Getting alteration statistics
                for k in range(n_alterations * batch_size):
                    orig_text = text[(i * batch_size) + k % batch_size]
                    try:
                        r_diffs, same_struct, syn_temp_diff, lex_temp_diff = \
                            self.get_sentence_statistics2(orig_text, alt_text[k],
                                                          orig_rels[(i * batch_size) + k % batch_size], alt_rels[k],
                                                          orig_temps[(i * batch_size) + k % batch_size], alt_temps[k])
                        sem_diffs = self.get_sem_stats(orig_sem[(i * batch_size) + k % batch_size], alt_sem[k])
                    except RecursionError or IndexError:
                        continue
                    stats.append([orig_text, alt_text[k], j, *(int(diff[0]) for diff in r_diffs),
                                  *(int(diff[1]) for diff in r_diffs),
                                  *(int(diff[2]) for diff in r_diffs), *(int(diff[0]) for diff in sem_diffs),
                                  *(int(diff[2]) for diff in sem_diffs),same_struct, syn_temp_diff, lex_temp_diff])

        header = ['original', 'altered', 'alteration_id', *(r + "_diff" for r in rels), *(r + "_struct" for r in rels),
                  *(r + "_ex" for r in rels), *(r + "_diff" for r in sems), *(r + "_ex" for r in sems),
                  'same_struct', 'syntemp_diff', 'lextemp_diff']
        df = pd.DataFrame(stats, columns=header)
        if not delta_drop:
            for diff_lab, struct_lab in zip([r+"_diff" for r in rels], [r+"_struct" for r in rels]):
                df[diff_lab] = [d or s for d, s in zip(df[diff_lab], df[struct_lab])]
        var_wise_scores = df.groupby('alteration_id').mean()[[r+"_diff" for r in rels]+['syntemp_diff', 'lextemp_diff']]
        var_wise_scores_struct = df.groupby('alteration_id').mean()[[r+"_struct" for r in rels]]
        var_wise_scores_ex = df.groupby('alteration_id').mean()[[r+"_ex" for r in rels]]
        var_wise_scores.set_axis([a.split('_')[0] for a in var_wise_scores.axes[1]], axis=1, inplace=True)
        if delta_drop:
            # renormalizing
            ex_array = np.array(var_wise_scores_ex)+1e-10
            ex_array = np.concatenate([ex_array, np.ones((sum(self.h_params.n_latents), 2))], axis=1)
            var_wise_scores = var_wise_scores/ex_array

        disent_score = 0
        lab_wise_disent = {}
        dec_disent_vars = {}
        for lab in rels:
            try:
                dec_disent_vars[lab] = var_wise_scores.idxmax()[lab]
            except TypeError:
                dec_disent_vars[lab] = -1
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

        # Getting Semantic role scores (default delta drop)
        sem_var_wise_scores = df.groupby('alteration_id').mean()[
            [r + "_diff" for r in sems]]
        sem_var_wise_scores_ex = df.groupby('alteration_id').mean()[[r + "_ex" for r in sems]]
        sem_var_wise_scores.set_axis([a.split('_')[0] for a in sem_var_wise_scores.axes[1]], axis=1, inplace=True)
        if delta_drop:
            # renormalizing
            ex_array = np.array(sem_var_wise_scores_ex) + 1e-10
            sem_var_wise_scores = sem_var_wise_scores / ex_array

        sem_disent_score, sem_lab_wise_disent, sem_dec_disent_vars = 0, {}, {}
        for lab in sems:
            try:
                sem_dec_disent_vars[lab] = sem_var_wise_scores.idxmax()[lab]
            except TypeError:
                sem_dec_disent_vars[lab] = -1
                sem_lab_wise_disent[lab] = 0
                continue
            top2 = np.array(sem_var_wise_scores.nlargest(2, lab)[lab])
            diff = top2[0] - top2[1]
            sem_lab_wise_disent[lab] = diff
            sem_disent_score += diff

        return disent_score, lab_wise_disent, var_wise_scores, dec_disent_vars, \
               sem_disent_score, sem_lab_wise_disent, sem_var_wise_scores, sem_dec_disent_vars

    def _get_stat_data_frame_all(self, n_samples=2000, n_alterations=1, batch_size=100, roles=None):
        roles = roles if roles is not None else['nsubj', 'verb', 'dobj', 'pobj']
        stats = []
        # Generating n_samples sentences
        text, samples, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                               sample_w=False, vary_z=True, complete=None)
        orig_rels = shallow_dependencies2(text, nlp, roles)
        orig_temps = truncated_template(text)
        for _ in tqdm(range(int(n_samples / batch_size)), desc="Generating original sentences"):
            text_i, samples_i, _ = self.get_sentences(n_samples=batch_size, gen_len=self.h_params.max_len - 1,
                                                       sample_w=False, vary_z=True, complete=None)
            text.extend(text_i)
            for k in samples.keys():
                samples[k] = torch.cat([samples[k], samples_i[k]])
            orig_rels.extend(shallow_dependencies2(text_i, nlp, roles))
            orig_temps.extend(truncated_template(text_i))
        for i in range(int(n_samples / batch_size)):
            for j in tqdm(range(sum(self.h_params.n_latents)), desc="Processing sample {}".format(str(i))):
                # Altering the sentences
                alt_text, _ = self._get_alternative_sentences(
                    prev_latent_vals={k: v[i * batch_size:(i + 1) * batch_size]
                                      for k, v in samples.items()},
                    params=None, var_z_ids=[j], n_samples=n_alterations,
                    gen_len=self.h_params.max_len - 1, complete=None)
                alt_rels = shallow_dependencies2(alt_text, nlp, roles)
                alt_temps = truncated_template(alt_text)
                # Getting alteration statistics
                for k in range(n_alterations * batch_size):
                    orig_text = text[(i * batch_size) + k % batch_size]
                    try:
                        diffs, same_struct, syn_temp_diff, lex_temp_diff = \
                            self.get_sentence_statistics_all(orig_text, alt_text[k],
                                                          orig_rels[(i * batch_size) + k % batch_size], alt_rels[k],
                                                          orig_temps[(i * batch_size) + k % batch_size], alt_temps[k],
                                                             roles)
                    except RecursionError or IndexError:
                        print("Rec err !")
                        continue
                    stats.append([orig_text, alt_text[k], j, *[int(diff[0]) for diff in diffs],
                                  *[int(diff[1]) for diff in diffs], same_struct, syn_temp_diff, lex_temp_diff])

        header = ['original', 'altered', 'alteration_id', *[r+'_diff' for r in roles], *[r+'_struct' for r in roles],
                  'same_struct', 'syntemp_diff',
                  'lextemp_diff']
        df = pd.DataFrame(stats, columns=header)
        var_wise_scores = df.groupby('alteration_id').mean()[[*[r+'_diff' for r in roles],
                                                              'syntemp_diff', 'lextemp_diff']]
        var_wise_scores_struct = df.groupby('alteration_id').mean()[[r+'_struct' for r in roles]]
        var_wise_scores.set_axis([a.split('_')[0] for a in var_wise_scores.axes[1]], axis=1, inplace=True)
        disent_score = 0
        lab_wise_disent = {}
        dec_disent_vars = {}
        for lab in roles:
            try:
                dec_disent_vars[lab] = var_wise_scores.idxmax()[lab]
            except TypeError :
                dec_disent_vars[lab] = -1
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

    def get_disentanglement_summaries_all(self, data_iter, n_samples=2000, roles=None):
        roles = roles if roles is not None else ["nsubj", "verb", "obj", "iobj"]
        with torch.no_grad():
            if self.h_params.graph_generator not in (get_vanilla_graph, get_vanilla_Transformer_graph):
                enc_var_wise_scores, enc_max_score, enc_lab_wise_disent, enc_disent_vars = \
                    self.get_encoder_disentanglement_score_all(data_iter, roles)
                self.writer.add_scalar('test/total_enc_disent_score', sum(enc_lab_wise_disent.values()), self.step)
                for k in enc_lab_wise_disent.keys():
                    self.writer.add_scalar('test/enc_disent_score[{}]'.format(k), enc_lab_wise_disent[k], self.step)
                enc_heatmap = get_hm_array2(pd.DataFrame(enc_var_wise_scores))#, "enc_heatmap_yelp.eps")
                self.writer.add_image('test/encoder_disentanglement', enc_heatmap, self.step)
                encoder_Ndisent_vars = len(set(enc_disent_vars.values()))
                self.writer.add_scalar('test/encoder_Ndisent_vars', encoder_Ndisent_vars, self.step)
            else:
                enc_lab_wise_disent, encoder_Ndisent_vars = {r: 0 for r in roles}, 0

            dec_disent_score, dec_lab_wise_disent, dec_var_wise_scores, dec_disent_vars\
                = self._get_stat_data_frame_all(n_samples=n_samples, roles=roles)
            self.writer.add_scalar('test/total_dec_disent_score', dec_disent_score, self.step)
            for k in dec_lab_wise_disent.keys():
                self.writer.add_scalar('test/dec_disent_score[{}]'.format(k), dec_lab_wise_disent[k], self.step)
            dec_heatmap = get_hm_array2(dec_var_wise_scores)#, "dec_heatmap_yelp.eps")
            self.writer.add_image('test/decoder_disentanglement', dec_heatmap, self.step)
            decoder_Ndisent_vars = len(set(dec_disent_vars.values()))
            self.writer.add_scalar('test/decoder_Ndisent_vars', decoder_Ndisent_vars, self.step)
        return dec_lab_wise_disent, enc_lab_wise_disent, decoder_Ndisent_vars, encoder_Ndisent_vars

    def get_disentanglement_summaries2(self, data_iter, n_samples=2000):
        with torch.no_grad():
            if self.h_params.graph_generator not in (get_vanilla_graph, get_vanilla_Transformer_graph):
                enc_var_wise_scores, enc_max_score, enc_lab_wise_disent, enc_disent_vars = \
                    self.get_encoder_disentanglement_score(data_iter)
                self.writer.add_scalar('test/total_enc_disent_score', sum(enc_lab_wise_disent.values()), self.step)
                enc_heatmap = get_hm_array2(pd.DataFrame(enc_var_wise_scores))#, "enc_heatmap_yelp.eps")
                self.writer.add_image('test/encoder_disentanglement', enc_heatmap, self.step)
                encoder_Ndisent_vars = len(set(enc_disent_vars.values()))
                self.writer.add_scalar('test/encoder_Ndisent_vars', encoder_Ndisent_vars, self.step)
                for k, v in enc_lab_wise_disent.items():
                    if np.isnan(v):
                        enc_lab_wise_disent[k] = 0
            else:
                enc_lab_wise_disent = {'nsubj': 0, 'verb': 0, 'dobj': 0, 'pobj': 0} if self.dataset != "de_nli" else\
                    {'sb': 0, 'verb': 0, 'oa': 0, 'da': 0, 'op': 0, 'oc': 0}
                encoder_Ndisent_vars = 0

            dec_disent_score, dec_lab_wise_disent, dec_var_wise_scores, dec_disent_vars\
                = self._get_stat_data_frame2(n_samples=n_samples)
            self.writer.add_scalar('test/total_dec_disent_score', dec_disent_score, self.step)
            dec_heatmap = get_hm_array2(dec_var_wise_scores)#, "dec_heatmap_yelp.eps")
            self.writer.add_image('test/decoder_disentanglement', dec_heatmap, self.step)
            decoder_Ndisent_vars = len(set(dec_disent_vars.values()))
            self.writer.add_scalar('test/decoder_Ndisent_vars', decoder_Ndisent_vars, self.step)


        return dec_lab_wise_disent, enc_lab_wise_disent,\
               decoder_Ndisent_vars, encoder_Ndisent_vars

    def get_disentanglement_summaries3(self, data_iter, n_samples=2000):
        with torch.no_grad():
            if self.h_params.graph_generator not in (get_vanilla_graph, get_vanilla_Transformer_graph):
                enc_var_wise_scores, enc_max_score, enc_lab_wise_disent, enc_disent_vars = \
                    self.get_encoder_disentanglement_score(data_iter)
                self.writer.add_scalar('test/total_enc_disent_score', sum(enc_lab_wise_disent.values()), self.step)
                enc_heatmap = get_hm_array2(pd.DataFrame(enc_var_wise_scores))#, "enc_heatmap_yelp.eps")
                self.writer.add_image('test/encoder_disentanglement', enc_heatmap, self.step)
                encoder_Ndisent_vars = len(set(enc_disent_vars.values()))
                self.writer.add_scalar('test/encoder_Ndisent_vars', encoder_Ndisent_vars, self.step)
                for k, v in enc_lab_wise_disent.items():
                    if np.isnan(v):
                        enc_lab_wise_disent[k] = 0
            else:
                enc_lab_wise_disent = {'nsubj': 0, 'verb': 0, 'dobj': 0, 'pobj': 0} if self.dataset != "de_nli" else\
                    {'sb': 0, 'verb': 0, 'oa': 0, 'da': 0, 'op': 0, 'oc': 0}
                encoder_Ndisent_vars = 0

            dec_disent_score, dec_lab_wise_disent, dec_var_wise_scores, dec_disent_vars, \
               sem_disent_score, sem_lab_wise_disent, sem_var_wise_scores, sem_dec_disent_vars\
                = self._get_stat_data_frame2(n_samples=n_samples)
            self.writer.add_scalar('test/total_dec_disent_score', dec_disent_score, self.step)
            dec_heatmap = get_hm_array2(dec_var_wise_scores)#, "dec_heatmap_yelp.eps")
            self.writer.add_image('test/decoder_disentanglement', dec_heatmap, self.step)
            decoder_Ndisent_vars = len(set(dec_disent_vars.values()))
            self.writer.add_scalar('test/decoder_Ndisent_vars', decoder_Ndisent_vars, self.step)

            self.writer.add_scalar('test/sem_dec_disent_score', sem_disent_score, self.step)
            sem_heatmap = get_hm_array2(sem_var_wise_scores)#, "dec_heatmap_yelp.eps")
            self.writer.add_image('test/sem_decoder_disentanglement', sem_heatmap, self.step)
            sem_decoder_Ndisent_vars = len(set(sem_dec_disent_vars.values()))
            self.writer.add_scalar('test/sem_decoder_Ndisent_vars', sem_decoder_Ndisent_vars, self.step)

            dec_disent_score_recon, dec_lab_wise_disent_recon, dec_var_wise_scores_recon, dec_disent_vars_recon, \
               sem_disent_score_recon, sem_lab_wise_disent_recon, sem_var_wise_scores_recon, sem_dec_disent_vars_recon\
                = self._get_stat_data_frame_recon(data_iter)
            self.writer.add_scalar('test/total_dec_disent_score_recon', dec_disent_score_recon, self.step)
            dec_heatmap_recon = get_hm_array2(dec_var_wise_scores_recon)#, "dec_heatmap_yelp.eps")
            self.writer.add_image('test/decoder_disentanglement_recon', dec_heatmap_recon, self.step)
            decoder_Ndisent_vars_recon = len(set(dec_disent_vars_recon.values()))
            self.writer.add_scalar('test/decoder_Ndisent_vars_recon', decoder_Ndisent_vars_recon, self.step)

            self.writer.add_scalar('test/sem_dec_disent_score_recon', sem_disent_score_recon, self.step)
            sem_heatmap_recon = get_hm_array2(sem_var_wise_scores_recon)#, "dec_heatmap_yelp.eps")
            self.writer.add_image('test/sem_decoder_disentanglement_recon', sem_heatmap_recon, self.step)
            sem_decoder_Ndisent_vars_recon = len(set(sem_dec_disent_vars_recon.values()))
            self.writer.add_scalar('test/sem_decoder_Ndisent_vars_recon', sem_decoder_Ndisent_vars_recon, self.step)

        return dec_lab_wise_disent, dec_lab_wise_disent_recon, enc_lab_wise_disent,\
               decoder_Ndisent_vars, decoder_Ndisent_vars_recon, encoder_Ndisent_vars

    def collect_stats(self, data_iter):
        kl, kl_var, rec, mi, nsamples = 0, 0, 0, 0, 0
        infer_prev, gen_prev = None, None
        loss_obj = self.losses[0]
        zs = [(self.infer_bn.name_to_v['z{}'.format(i+1)], self.gen_bn.name_to_v['z{}'.format(i+1)])
              for i in range(len(self.h_params.n_latents))]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_iter, desc="Getting Model Stats")):
                if batch.text.shape[1] < 2: continue
                inp = {'x': batch.text[..., 1:], 'x_prev': batch.text[..., :-1]}
                if self.h_params.sup_coeff > 0 or self.h_params.dec_sup_coeff > 0:
                    inp['sup'] = batch.label
                infer_prev, gen_prev = self(inp, prev_states=(infer_prev, gen_prev))
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
def batch_sent_relations(sents):
    target = [{'sentence': sent} for sent in sents]
    preds = predictor.predict_batch_json(target)
    sent_dicts = []
    for pred in preds:
        sent_dict = []
        for el in pred['verbs']:
            sent_dict.append({})
            for v_i in el['description'].split('[')[1:]:
                in_bracket = v_i.split(']')[0]
                try:
                    arg_l, arg_str = in_bracket.split(':')
                    sent_dict[-1][arg_l] = arg_str
                except ValueError as e:
                    print('this raised an anomaly:', el)
        sent_dicts.append(sent_dict)
    return sent_dicts


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
    snsplt = sns.heatmap(df, cmap ='Reds', linewidths = 0.20, annot=True)
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


def shallow_dependencies_pos(sents, roles=None):
    # this one is for PoS tags
    docs = nlp.pipe(sents)
    relations = []
    for doc in docs:
        realizations = {r: ['', []] for r in roles}
        for i, tok in enumerate(doc):
            for r in roles:
                if tok.pos_ == r and realizations[r][0] == '':
                        realizations[r] = [tok.text, [tok.i]]

        relations.append({'text':{k: v[0] for k, v in realizations.items()},
                         'idx': {k: v[1] for k, v in realizations.items()}})
    return relations


def shallow_dependencies2(sents, parser, roles=None):
    roles = roles if roles is not None else['nsubj', 'verb', 'dobj', 'pobj']
    docs = parser.pipe(sents)
    relations = []
    for doc in docs:
        realizations = {r: ['', []] for r in roles}
        for i, tok in enumerate(doc):
            for r in roles:
                if r == 'verb':
                    if tok.dep_ == 'ROOT' and tok.pos_ == 'VERB':
                        realizations[r] = [tok.text, [tok.i]]
                else:
                    expected_ancestors = 2 if r == 'pobj' else 1
                    if len(list(tok.ancestors)) != expected_ancestors:
                        continue
                    parent = list(tok.ancestors)[expected_ancestors-1]
                    parent_is_verbal_root = parent.dep_ == 'ROOT' and parent.pos_ == 'VERB'
                    if tok.dep_ == r and realizations[r][0] == '' and parent_is_verbal_root:
                        realizations[r] = [' '.join([toki.text for toki in tok.subtree]),
                                           [toki.i for toki in tok.subtree]]

        relations.append({'text':{k: v[0] for k, v in realizations.items()},
                         'idx': {k: v[1] for k, v in realizations.items()}})
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


def get_srl_from_prs(pr, doc):
    roles = ['V', 'ARG0', 'ARG1']
    realizations = {r: ['', []] for r in roles}
    root, desc = None, None
    for tok in doc:
        if tok.dep_ == 'ROOT':
            root = tok.i
            break
    if root is None: return False, realizations
    for v in pr["verbs"]:
        v_idx = [i for i in range(len(v['tags'])) if v['tags'][i].endswith('V')]
        if root in v_idx:
            for k in realizations.keys():
                k_idx = [i for i in range(len(v['tags'])) if v['tags'][i].endswith(k)]
                realizations[k] = [' '.join([pr['words'][j] for j in k_idx]), k_idx]
            return True, realizations
    return False, realizations
    # relations.append({'text':{k: v[0] for k, v in realizations.items()},
    #                      'idx': {k: v[1] for k, v in realizations.items()}})

def srl_from_sents(sents):
    relations = []
    docs = nlp.pipe(sents)
    prs = predictor.predict_batch_json([{'sentence': sent} for sent in sents])
    for pr, doc in zip(prs, docs):
        _, realizations = get_srl_from_prs(pr, doc)
        relations.append({'text':{k: v[0] for k, v in realizations.items()},
                             'idx': {k: v[1] for k, v in realizations.items()}})
    return relations
