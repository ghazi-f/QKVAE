# This file will implement the main training loop for a model
from time import time
import argparse

from torch import device
import torch
import numpy as np

from data_prep import UDPoSDaTA as Data
from pos_tagging.models import SSPoSTag as Model
from pos_tagging.h_params import DefaultSSPoSTagHParams as HParams
from pos_tagging.vertices import *
from components.criteria import *
parser = argparse.ArgumentParser()

# Training and Optimization
parser.add_argument("--test_name", default='unnamed', type=str)
parser.add_argument("--max_len", default=32, type=int)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--grad_accu", default=8, type=int)
parser.add_argument("--n_epochs", default=100, type=int)
parser.add_argument("--test_freq", default=16, type=int)
parser.add_argument("--complete_test_freq", default=16, type=int)
parser.add_argument("--supervision_proportion", default=1., type=float)
parser.add_argument("--device", default='cuda:0', choices=["cuda:0", "cuda:1", "cuda:2", "cpu"], type=str)
parser.add_argument("--embedding_dim", default=200, type=int)
parser.add_argument("--pos_embedding_dim", default=50, type=int)
parser.add_argument("--z_size", default=100, type=int)
parser.add_argument("--text_rep_l", default=2, type=int)
parser.add_argument("--text_rep_h", default=500, type=int)
parser.add_argument("--encoder_h", default=500, type=int)
parser.add_argument("--encoder_l", default=1, type=int)
parser.add_argument("--pos_h", default=400, type=int)
parser.add_argument("--pos_l", default=1, type=int)
parser.add_argument("--decoder_h", default=500, type=int)
parser.add_argument("--decoder_l", default=2, type=int)
parser.add_argument("--highway", default=False, type=bool)
parser.add_argument("--losses", default='SSVAE', choices=["S", "SSVAE", "SSPIWO", "SSIWAE"], type=str)
parser.add_argument("--training_iw_samples", default=5 , type=int)
parser.add_argument("--testing_iw_samples", default=20, type=int)
parser.add_argument("--test_prior_samples", default=5, type=int)
parser.add_argument("--anneal_kl0", default=5000, type=int)
parser.add_argument("--anneal_kl1", default=25000, type=int)
parser.add_argument("--grad_clip", default=10., type=float)
parser.add_argument("--kl_th", default=None, type=float or None)
parser.add_argument("--dropout", default=0.33, type=float)
parser.add_argument("--lr", default=2e-3, type=float)

flags = parser.parse_args()

# Manual Settings, Deactivate before pushing
if False:
    flags.losses = 'SSVAE'
    flags.batch_size = 10
    flags.grad_accu = 8
    flags.test_name = "SSVAE/0.1test"
    flags.supervision_proportion = 0.1

MAX_LEN = flags.max_len
BATCH_SIZE = flags.batch_size
GRAD_ACCU = flags.grad_accu
N_EPOCHS = flags.n_epochs
TEST_FREQ = flags.test_freq
COMPLETE_TEST_FREQ = flags.complete_test_freq
SUP_PROPORTION = flags.supervision_proportion
DEVICE = device(flags.device)
LOSSES = {'S': [Supervision],
          'SSVAE': [Supervision, ELBo],
          'SSPIWO': [Supervision, IWLBo],
          'SSIWAE': [Supervision, IWLBo]}[flags.losses]
ANNEAL_KL = [flags.anneal_kl0, flags.anneal_kl1] if flags.losses != 'S' else [0, 0]
# Changed loss params right after the beginning of SSVAE Exps
LOSS_PARAMS = [1] if flags.losses == 'S' else [1, 1]
PIWO = flags.losses == 'SSPIWO'


def main():
    data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE)
    h_params = HParams(len(data.vocab.itos), len(data.tags.itos), MAX_LEN, BATCH_SIZE, N_EPOCHS,
                       device=DEVICE, pos_ignore_index=data.tags.stoi['<pad>'],
                       vocab_ignore_index=data.vocab.stoi['<pad>'], decoder_h=flags.decoder_h,
                       decoder_l=flags.decoder_l, encoder_h=flags.encoder_h, encoder_l=flags.encoder_l,
                       text_rep_h=flags.text_rep_h, text_rep_l=flags.text_rep_l,
                       test_name=flags.test_name, grad_accumulation_steps=GRAD_ACCU,
                       optimizer_kwargs={'lr': flags.lr/GRAD_ACCU, 'weight_decay': 0.0, 'betas': (0.9, 0.9)},
                       is_weighted=[], graph_generator=get_graph_postag, z_size=flags.z_size,
                       embedding_dim=flags.embedding_dim, pos_embedding_dim=flags.pos_embedding_dim, pos_h=flags.pos_h,
                       pos_l=flags.pos_l, anneal_kl=ANNEAL_KL, grad_clip=flags.grad_clip,
                       kl_th=flags.kl_th, highway=flags.highway, losses=LOSSES, dropout=flags.dropout,
                       training_iw_samples=flags.training_iw_samples, testing_iw_samples=flags.testing_iw_samples,
                       loss_params=LOSS_PARAMS, piwo=PIWO)
    val_iterator = iter(data.val_iter)
    supervised_iterator = iter(data.sup_iter)
    print("Words: ", len(data.vocab.itos), ", Target tags: ", len(data.tags.itos), ", On device: ", DEVICE.type)
    print("Loss Type: ", flags.losses, ", Supervision proportion: ", SUP_PROPORTION)
    model = Model(data.vocab, data.tags, h_params, wvs=data.wvs)
    if DEVICE.type == 'cuda':
        model.cuda(DEVICE)

    total_train_samples = len(data.train_iter.dataset.examples)
    current_time = time()
    #print(model)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    max_acc = 0
    wait_count = 0
    waiting_epochs = 5
    sup_samples_count = 0
    loss = torch.tensor(1e20)

    while data.train_iter is not None:
        for training_batch in data.train_iter:
            if len(training_batch.text[0]) > MAX_LEN:
                training_batch.text = training_batch.text[:, :MAX_LEN]
                training_batch.text = training_batch.label[:, :MAX_LEN-2]
            if model.step == h_params.anneal_kl[0]:
                model.optimizer = h_params.optimizer(model.parameters(), **h_params.optimizer_kwargs)
                print('Refreshed optimizer !')
                if model.step != 0 and not torch.isnan(loss):
                    model.save()
                    print('Saved model after it\'s pure reconstruction phase')

            supervised_batch = limited_next(supervised_iterator)
            #print('heyheyhey', supervised_batch.text.shape, supervised_batch.label.shape)
            '''print([' '.join(['('+data.vocab.itos[t]+' '+data.tags.itos[l]+')' for t, l in zip(text_i[1:], lab_i)]) for
                   text_i, lab_i in zip(supervised_batch.text[:2], supervised_batch.label[:2])])'''

            loss = model.opt_step({'x': training_batch.text}) if flags.losses != 'S' else 0
            loss += model.opt_step({'x': supervised_batch.text, 'y': supervised_batch.label})
            sup_samples_count += BATCH_SIZE

            print("step:{}, loss:{}, seconds/step:{}".format(model.step, loss, time()-current_time))
            if int(model.step/ (1 if flags.losses == 'S' else 2)) % TEST_FREQ == TEST_FREQ-1:
                print('haha')
                model.eval()
                try:
                    test_batch = limited_next(val_iterator)
                except StopIteration:
                    print("Reinitialized test data iterator")
                    val_iterator = iter(data.val_iter)
                    test_batch = limited_next(val_iterator)
                with torch.no_grad():
                    model({'x': test_batch.text, 'y': test_batch.label})
                model.dump_test_viz(complete=int(model.step/ (1 if flags.losses == 'S' else 2)) %
                                    COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)
                model.train()

            current_time = time()
            if sup_samples_count >= (total_train_samples * SUP_PROPORTION):
                print("Reinitialized supervised training iterator")
                supervised_iterator = iter(data.sup_iter)
                sup_samples_count = 0

        data.reinit_iterator('valid')
        if model.step > h_params.anneal_kl[0]:
            model.eval()
            if model.generate:
                model.get_perplexity(data.val_iter)
                data.reinit_iterator('valid')
            accuracy = model.get_overall_accuracy(data.val_iter)
            if accuracy > max_acc:
                print('Saving The model ..')
                max_acc = accuracy
                model.save()
                wait_count = 0
            else:
                wait_count += 1

            if wait_count == waiting_epochs:
                model.reduce_lr(10.)
                print('Learning rate reduced to ', [gr['lr'] for gr in model.optimizer.param_groups])

            if wait_count == waiting_epochs*2:
                break

            model.train()
        data.reinit_iterator('valid')
        data.reinit_iterator('train')
    print("Finished training")


def limited_next(iterator):
    batch = next(iterator)
    if len(batch.text[0]) > MAX_LEN:
        batch.text = batch.text[:, :MAX_LEN]
        batch.label = batch.label[:, :MAX_LEN-2]
    return batch


if __name__ == '__main__':
    main()


