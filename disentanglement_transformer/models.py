from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from disentanglement_transformer.h_params import *
from components.bayesnets import BayesNet
from components.criteria import Supervision
from components.latent_variables import MultiCategorical
import spacy
from allennlp.predictors.predictor import Predictor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks", {"xtick.major.color": 'white', "ytick.major.color": 'white'})

nlp = spacy.load("en_core_web_sm")

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")


# ==================================================== SSPOSTAG MODEL CLASS ============================================

class DisentanglementTransformerVAE(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, vocab_index, tag_index, h_params, autoload=True, wvs=None):
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

        #                          ----------- Unsupervised Forward/Backward ----------------
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

        # Loss computation and backward pass
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
                                            for w in sen])#.split('<eos>')[0]
                                  for sen in x_hat_params]).replace('<pad>', '_').replace('_unk', '<?>').replace('<eos>', '\n')
        else:
            samples = [' '.join([self.index[self.generated_v].itos[w]
                                 for w in sen]).split('<eos>')[0].replace('<go>', '').replace('</go>', '')
                       for sen in x_hat_params]
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
                             for w in sen]).split('<eos>')[0].replace('<go>', '').replace('</go>', '')
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

    def get_disentanglement_summaries(self):
        df = self._get_stat_data_frame(n_samples=40, n_alterations=10, batch_size=20)
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
            largest2 = np.array(grouped.mean()[d_rel_types].transpose().nlargest(2, i)[i].array)
            sup_dis_diffs1 += largest2[0] - largest2[1]
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


def get_depth(root, toks, tree, depth=0):
    root_tree = list([tok for tok in tree[root]])
    if len(root_tree) > 0:
        child_ids = [i for i, tok in enumerate(toks) if tok in root_tree]
        return 1 + max([get_depth(child_id, toks, tree) for child_id in child_ids])
    else:
        return depth


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

