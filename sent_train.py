# This file will implement the main training loop for a model
from time import time
import argparse
import os

from torch import device
import torch
from torch import optim
import numpy as np

from data_prep import HuggingIMDB2, HuggingAGNews, HuggingYelp
from sentence_classification.models import SSSentenceClassification as Model
from sentence_classification.h_params import DefaultSSSentenceClassificationHParams as HParams
from sentence_classification.graphs import *
from components.criteria import *
parser = argparse.ArgumentParser()

# Training and Optimization
parser.add_argument("--test_name", default='unnamed', type=str)
parser.add_argument("--dataset", default='imdb', choices=["imdb", "ag_news", "yelp"], type=str)
parser.add_argument("--result_csv", default='imdb.csv', type=str)
parser.add_argument("--max_len", default=128, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--grad_accu", default=2, type=int)
parser.add_argument("--n_epochs", default=10000, type=int)
parser.add_argument("--test_freq", default=32, type=int)
parser.add_argument("--complete_test_freq", default=160, type=int)
parser.add_argument("--supervision_proportion", default=1., type=float)
parser.add_argument("--dev_index", default=1, type=float)
parser.add_argument("--unsupervision_proportion", default=1., type=float)
parser.add_argument("--generation_weight", default=1, type=float)
parser.add_argument("--device", default='cuda:0', choices=["cuda:0", "cuda:1", "cuda:2", "cpu"], type=str)
parser.add_argument("--embedding_dim", default=500, type=int)
parser.add_argument("--pos_embedding_dim", default=100, type=int)
parser.add_argument("--z_size", default=200, type=int)
parser.add_argument("--text_rep_l", default=2, type=int)
parser.add_argument("--text_rep_h", default=500, type=int)
parser.add_argument("--encoder_h", default=500, type=int)
parser.add_argument("--encoder_l", default=1, type=int)
parser.add_argument("--pos_h", default=100, type=int)
parser.add_argument("--pos_l", default=1, type=int)
parser.add_argument("--decoder_h", default=500, type=int)
parser.add_argument("--decoder_l", default=1, type=int)
parser.add_argument("--highway", default=False, type=bool)
parser.add_argument("--markovian", default=True, type=bool)
parser.add_argument("--losses", default='SSVAE', choices=["S", "VAE", "SSVAE", "SSPIWO", "SSiPIWO", "SSIWAE"], type=str)
parser.add_argument("--training_iw_samples", default=10, type=int)
parser.add_argument("--testing_iw_samples", default=1, type=int)
parser.add_argument("--test_prior_samples", default=2, type=int)
parser.add_argument("--anneal_kl0", default=3000, type=int)
parser.add_argument("--anneal_kl1", default=6000, type=int)
parser.add_argument("--grad_clip", default=5., type=float)
parser.add_argument("--kl_th", default=0., type=float or None)
parser.add_argument("--dropout", default=0., type=float)
parser.add_argument("--word_dropout", default=0.0, type=float)
parser.add_argument("--l2_reg", default=0., type=float)
parser.add_argument("--lr", default=2e-3, type=float)
parser.add_argument("--lr_reduction", default=4., type=float)
parser.add_argument("--wait_epochs", default=4, type=float)
parser.add_argument("--stopping_crit", default="convergence", choices=["convergence", "early"], type=str)

flags = parser.parse_args()
# Manual Settings, Deactivate before pushing
if False:
    flags.losses = 'SSPIWO'
    flags.batch_size = 8
    flags.grad_accu = 8
    flags.max_len = 128
    flags.test_name = "SSVAE/IMDB/test5"
    flags.supervision_proportion = 1

# torch.autograd.set_detect_anomaly(True)
Data = {'imdb': HuggingIMDB2, 'ag_news': HuggingAGNews, 'yelp': HuggingYelp}[flags.dataset]
MAX_LEN = flags.max_len
BATCH_SIZE = flags.batch_size
GRAD_ACCU = flags.grad_accu
N_EPOCHS = flags.n_epochs
TEST_FREQ = flags.test_freq
COMPLETE_TEST_FREQ = flags.complete_test_freq
SUP_PROPORTION = flags.supervision_proportion
DEV_INDEX = flags.dev_index
UNSUP_PROPORTION = flags.unsupervision_proportion
DEVICE = device(flags.device)
# This prevents illegal memory access on multigpu machines (unresolved issue on torch's github)
if flags.device.startswith('cuda'):
    torch.cuda.set_device(int(flags.device[-1]))
LOSSES = {'S': [Supervision],
          'SSVAE': [Supervision, ELBo],
          'SSPIWO': [Supervision, IWLBo],
          'SSiPIWO': [Supervision, IWLBo],
          'SSIWAE': [Supervision, IWLBo],
          'VAE': [ELBo],
          'Reconstruction': [Reconstruction]}[flags.losses]
#  LOSSES = [IWLBo]
ANNEAL_KL = [flags.anneal_kl0*flags.grad_accu, flags.anneal_kl1*flags.grad_accu] if flags.losses != 'S' else [0, 0]
LOSS_PARAMS = [1] if 'SS' not in flags.losses else [1, flags.generation_weight]
if flags.grad_accu > 1:
    LOSS_PARAMS = [w/flags.grad_accu for w in LOSS_PARAMS]
PIWO = flags.losses == 'SSPIWO'
IPIWO = flags.losses == 'SSiPIWO'


def main():
    data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE, UNSUP_PROPORTION, SUP_PROPORTION, DEV_INDEX)
    h_params = HParams(len(data.vocab.itos), len(data.tags.itos), MAX_LEN, BATCH_SIZE, N_EPOCHS,
                       device=DEVICE, pos_ignore_index=data.tags.stoi['<pad>'],
                       vocab_ignore_index=data.vocab.stoi['<pad>'], decoder_h=flags.decoder_h,
                       decoder_l=flags.decoder_l, encoder_h=flags.encoder_h, encoder_l=flags.encoder_l,
                       text_rep_h=flags.text_rep_h, text_rep_l=flags.text_rep_l,
                       test_name=flags.test_name, grad_accumulation_steps=GRAD_ACCU,
                       optimizer_kwargs={'lr': flags.lr, #'weight_decay': flags.l2_reg, 't0':100, 'lambd':0.},
                                         'weight_decay': flags.l2_reg, 'betas': (0.9, 0.85)},
                       is_weighted=[], graph_generator=get_sentiment_graph, z_size=flags.z_size,
                       embedding_dim=flags.embedding_dim, pos_embedding_dim=flags.pos_embedding_dim, pos_h=flags.pos_h,
                       pos_l=flags.pos_l, anneal_kl=ANNEAL_KL, grad_clip=flags.grad_clip*flags.grad_accu,
                       kl_th=flags.kl_th, highway=flags.highway, losses=LOSSES, dropout=flags.dropout,
                       training_iw_samples=flags.training_iw_samples, testing_iw_samples=flags.testing_iw_samples,
                       loss_params=LOSS_PARAMS, piwo=PIWO, ipiwo=IPIWO, optimizer=optim.AdamW, markovian=flags.markovian
                       , word_dropout=flags.word_dropout, contiguous_lm=False)
    val_iterator = iter(data.val_iter)
    supervised_iterator = iter(data.sup_iter)
    print("Launching experiment ", flags.test_name)
    print("Words: ", len(data.vocab.itos), ", Target tags: ", len(data.tags.itos), ", On device: ", DEVICE.type)
    print("Loss Type: ", flags.losses, ", Supervision proportion: ", SUP_PROPORTION)
    model = Model(data.vocab, data.tags, h_params, wvs=data.wvs)
    if DEVICE.type == 'cuda':
        model.cuda(DEVICE)

    total_unsupervised_train_samples = len(data.train_iter)*BATCH_SIZE
    total_supervised_train_samples = len(data.sup_iter.dataset.examples)
    print("Unsupervised training examples: ", total_unsupervised_train_samples,
          ", Supervised training examples: ", total_supervised_train_samples, len(data.val_iter.dataset.examples))
    current_time = time()
    #print(model)
    number_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.infer_bn.parameters() if p.requires_grad)
    print("Inference parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.gen_bn.parameters() if p.requires_grad)
    print("Generation parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.word_embeddings.parameters() if p.requires_grad)
    print("Embedding parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    max_acc = 0
    min_perp = 1e20
    wait_count = 0
    loss = torch.tensor(1e20)
    mean_loss = 0
    supervision_epoch = 0
    best_epoch = -1
    while data.train_iter is not None:
        for i, training_batch in enumerate(data.train_iter):
            if training_batch.text.shape[1] < 2:
                print('Misshaped training sample')
                continue

            '''print([' '.join([data.vocab.itos[t] for t in text_i]) for text_i in training_batch.text[:2]])'''

            if model.step == h_params.anneal_kl[0]:
                model.optimizer = h_params.optimizer(model.parameters(), **h_params.optimizer_kwargs)
                print('Refreshed optimizer !')
                if model.step != 0 and not torch.isnan(loss):
                    model.save()
                    print('Saved model after it\'s pure reconstruction phase')
            try:
                supervised_batch = next(supervised_iterator)
            except StopIteration:
                print("Reinitialized supervised training iterator")
                supervision_epoch += 1
                supervised_iterator = iter(data.sup_iter)
                if 'S' in flags.losses and model.step >= h_params.anneal_kl[0]:
                    model.eval()
                    accuracy_split = data.val_iter if flags.stopping_crit == "early" else data.sup_iter
                    accuracy = model.get_overall_accuracy(accuracy_split)
                    model.train()
                    print("Accuracy is {} at step {}".format(accuracy, model.step))
                    if accuracy > max_acc:
                        print('Saving The model ..')
                        max_acc = accuracy
                        model.save()
                        wait_count = 0
                        best_epoch = supervision_epoch
                    else:
                        wait_count += 1
                    if wait_count == flags.wait_epochs:
                        model.reduce_lr(flags.lr_reduction)
                        print('Learning rate reduced to ', [gr['lr'] for gr in model.optimizer.param_groups])

                    if wait_count == flags.wait_epochs * 2:
                        break

            """print([' '.join(['('+data.vocab.itos[t]+' '+data.tags.itos[l]+')' for t, l in zip(text_i[1:], lab_i)]) for
                   text_i, lab_i in zip(supervised_batch.text[:2], supervised_batch.label[:2])])"""

            loss = model.opt_step({'x': training_batch.text[..., 1:], 'x_prev': training_batch.text[..., :-1]}) if flags.losses != 'S' else 0
            loss += model.opt_step({'x': supervised_batch.text[..., 1:], 'x_prev': supervised_batch.text[..., :-1],
                                    'y': supervised_batch.label}) if 'S' in flags.losses \
                else 0

            mean_loss += loss
            if i % 30 == 0:
                mean_loss /= 30
                print("step:{}, loss:{}, seconds/step:{}".format(model.step, mean_loss, time()-current_time))
                mean_loss = 0

            if int(model.step / (len(LOSSES))) % TEST_FREQ == TEST_FREQ-1:
                model.eval()
                try:
                    val_batch = limited_next(val_iterator)
                except StopIteration:
                    val_iterator = iter(data.val_iter)
                    print("Reinitialized test data iterator")
                    val_batch = limited_next(val_iterator)
                with torch.no_grad():
                    model({'x': val_batch.text[..., 1:], 'x_prev': val_batch.text[..., :-1], 'y': val_batch.label})
                model.dump_test_viz(complete=int(model.step / (len(LOSSES))) %
                                    COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)
                model.train()

            current_time = time()
        data.reinit_iterator('valid')
        if model.step >= h_params.anneal_kl[0]:
            model.eval()
            if 'S' not in flags.losses:
                pp_ub = model.get_perplexity(data.unsup_val_iter)
                print("Perplexity Upper Bound is {} at step {}".format(pp_ub, model.step))
                data.reinit_iterator('valid')
                if pp_ub < min_perp:
                    print('Saving The model ..')
                    min_perp = pp_ub
                    model.save()
                    wait_count = 0
                else:
                    wait_count += 1

                if wait_count == flags.wait_epochs:
                    model.reduce_lr(flags.lr_reduction)
                    print('Learning rate reduced to ', [gr['lr'] for gr in model.optimizer.param_groups])

            if wait_count == flags.wait_epochs*2:
                break

            model.train()
        data.reinit_iterator('valid')
        data.reinit_iterator('unsup_valid')
        data.reinit_iterator('train')
    print("Finished training, starting final evaluation :")
    # Reloading best parameters
    model.load()
    # Getting final test numbers

    model.eval()
    test_accuracy = model.get_overall_accuracy(data.test_iter).item()
    if 'SS' in flags.losses:
        pp_ub = model.get_perplexity(data.test_iter).item()
    else:
        pp_ub = -1
    print("Final Test Accuracy is: {}, Final test perplexity is: {}".format(test_accuracy, pp_ub))
    if not os.path.exists(flags.result_csv):
        with open(flags.result_csv, 'w') as f:
            f.write(', '.join(['test_name', 'dev_index', 'loss_type', 'supervision_proportion', 'generation_weight'
                               'unsupervision_proportion', 'test_accuracy', 'dev_accuracy', 'pp_ub', 'best_epoch',
                               'embedding_dim', 'pos_embedding_dim', 'z_size',
                               'text_rep_l', 'text_rep_h', 'encoder_h', 'encoder_l',
                               'pos_h', 'pos_l', 'decoder_h', 'decoder_l',
                               ]) + '\n')

    with open(flags.result_csv, 'a') as f:
        f.write(', '.join([flags.test_name, str(flags.dev_index), flags.losses, str(flags.supervision_proportion),
                           str(flags.generation_weight),
                           str(flags.unsupervision_proportion), str(test_accuracy), str(max_acc.item()), str(pp_ub), str(best_epoch),
                           str(flags.embedding_dim), str(flags.pos_embedding_dim), str(flags.z_size),
                           str(flags.text_rep_l), str(flags.text_rep_h), str(flags.encoder_h), str(flags.encoder_l),
                           str(flags.pos_h), str(flags.pos_l), str(flags.decoder_h), str(flags.decoder_l),
                           ])+'\n')


def limited_next(iterator):
    batch = next(iterator)
    if len(batch.text[0]) > MAX_LEN:
        batch.text = batch.text[:, :MAX_LEN]
        batch.label = batch.label[:, :MAX_LEN-1]
    return batch


if __name__ == '__main__':
    main()


