# This file will implement the main training loop for a model
from time import time
import argparse

from torch import device
import torch
from torch import optim
import numpy as np

from data_prep import LexNorm2015Data
from normalization.models import UnsupervisedTrainingHandler, DistantlySupervisedTrainingHandler, \
    SupervisedTrainingHandler, SemiSupervisedTrainingHandler
from normalization.h_params import DefaultTransformerHParams as HParams
from normalization.graphs import get_normalization_graphs
from components.criteria import *
parser = argparse.ArgumentParser()
from torch.nn import MultiheadAttention
# Training and Optimization
parser.add_argument("--test_name", default='unnamed', type=str)
parser.add_argument("--data", default='lexnorm2015', choices=["lexnorm2015"], type=str)
parser.add_argument("--w_max_len", default=30, type=int)
parser.add_argument("--c_max_len", default=10, type=int)
parser.add_argument("--w_max_vocab", default=2000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--sup_proportion", default=1., type=float)
parser.add_argument("--grad_accu", default=1, type=int)
parser.add_argument("--n_epochs", default=10000, type=int)
parser.add_argument("--test_freq", default=32, type=int)
parser.add_argument("--complete_test_freq", default=160, type=int)
parser.add_argument("--device", default='cuda:0', choices=["cuda:0", "cuda:1", "cuda:2", "cpu"], type=str)
parser.add_argument("--w_embedding_dim", default=400, type=int)
parser.add_argument("--c_embedding_dim", default=200, type=int)
parser.add_argument("--y_embedding_dim", default=120, type=int)
parser.add_argument('--pretrained_embeddings', dest='pretrained_embeddings', action='store_true')
parser.add_argument('--no-pretrained_embeddings', dest='pretrained_embeddings', action='store_false')
parser.set_defaults(pretrained_embeddings=False)
parser.add_argument("--zcom_size", default=250, type=int)
parser.add_argument("--zdiff_size", default=250, type=int)
parser.add_argument("--c_encoder_h", default=200, type=int)
parser.add_argument("--c_encoder_l", default=2, type=int)
parser.add_argument("--c_decoder_h", default=100, type=int)
parser.add_argument("--c_decoder_l", default=2, type=int)
parser.add_argument("--w_encoder_h", default=400, type=int)
parser.add_argument("--w_encoder_l", default=2, type=int)
parser.add_argument("--w_decoder_h", default=400, type=int)
parser.add_argument("--w_decoder_l", default=2, type=int)
parser.add_argument("--y_encoder_h", default=30, type=int)
parser.add_argument("--y_encoder_l", default=1, type=int)
parser.add_argument("--mode", default='unsupervised', choices=["unsupervised", "distantly_supervised", "supervised",
                                                             "semi_supervised"], type=str)
parser.add_argument("--testing_iw_samples", default=20, type=int)
parser.add_argument("--anneal_kl0", default=2000, type=int)
parser.add_argument("--anneal_kl1", default=4000, type=int)
parser.add_argument("--grad_clip", default=100., type=float)
parser.add_argument("--kl_th", default=0/(768/2), type=float or None)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--word_dropout", default=.0, type=float)
parser.add_argument("--char_dropout", default=.0, type=float)
parser.add_argument("--l2_reg", default=0, type=float)
parser.add_argument("--lr", default=4e-3, type=float)
parser.add_argument("--lr_reduction", default=4., type=float)
parser.add_argument("--wait_epochs", default=1, type=float)
parser.add_argument("--save_all", default=True, type=bool)

flags = parser.parse_args()

# Manual Settings, Deactivate before pushing
if True:
    flags.batch_size = 32
    flags.test_name = "normalization/testunsup1"

# torch.autograd.set_detect_anomaly(True)
GRAPH = get_normalization_graphs
Data = {"lexnorm2015": LexNorm2015Data}[flags.data]
Model = {"unsupervised": UnsupervisedTrainingHandler,
         "distantly_supervised": DistantlySupervisedTrainingHandler,
         "supervised": SupervisedTrainingHandler,
         "semi_supervised": SemiSupervisedTrainingHandler}[flags.mode]
C_MAX_LEN = flags.c_max_len
W_MAX_LEN = flags.w_max_len
BATCH_SIZE = flags.batch_size
GRAD_ACCU = flags.grad_accu
N_EPOCHS = flags.n_epochs
TEST_FREQ = flags.test_freq
COMPLETE_TEST_FREQ = flags.complete_test_freq
DEVICE = device(flags.device)
# This prevents illegal memory access on multigpu machines (unresolved issue on torch's github)
if flags.device.startswith('cuda'):
    torch.cuda.set_device(int(flags.device[-1]))
#  LOSSES = [IWLBo]
ANNEAL_KL = [flags.anneal_kl0*flags.grad_accu, flags.anneal_kl1*flags.grad_accu]

def main():
    data = Data(W_MAX_LEN, C_MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE, flags.sup_proportion, flags.mode, flags.w_max_vocab)
    val_iterator = iter(data.val_iter)
    print("Words: ", len(data.w_vocab.itos), ", Characters: ", len(data.c_vocab.itos), ", On device: ", DEVICE.type)
    h_params = HParams(len(data.c_vocab.itos), len(data.w_vocab.itos), C_MAX_LEN, W_MAX_LEN, BATCH_SIZE, N_EPOCHS,
                       device=DEVICE, c_ignore_index=data.c_vocab.stoi['<pad>'],
                       w_ignore_index=data.w_vocab.stoi['<pad>'], y_ignore_index=None,
                       c_decoder_h=flags.c_decoder_h, w_decoder_h=flags.w_decoder_h,
                       c_decoder_l=flags.c_decoder_l, w_decoder_l=flags.w_decoder_l,
                       c_encoder_h=flags.c_encoder_h, w_encoder_h=flags.w_encoder_h, y_encoder_h=flags.y_encoder_h,
                       c_encoder_l=flags.c_encoder_l, w_encoder_l=flags.w_encoder_l, y_encoder_l=flags.y_encoder_l,
                       test_name=flags.test_name, grad_accumulation_steps=GRAD_ACCU,
                       optimizer_kwargs={'lr': flags.lr, #'weight_decay': flags.l2_reg, 't0':100, 'lambd':0.},
                                         'weight_decay': flags.l2_reg, 'betas': (0.9, 0.85)},
                       is_weighted=[], graph_generator=GRAPH,
                       zcom_size=flags.zcom_size, zdiff_size=flags.zdiff_size,
                       w_embedding_dim=flags.w_embedding_dim, c_embedding_dim=flags.c_embedding_dim,
                       y_embedding_dim=flags.y_embedding_dim, anneal_kl=ANNEAL_KL,
                       grad_clip=flags.grad_clip*flags.grad_accu, kl_th=flags.kl_th, dropout=flags.dropout,
                       testing_iw_samples=flags.testing_iw_samples, optimizer=optim.AdamW,
                       word_dropout=flags.word_dropout, contiguous_lm=False
                       )
    model = Model(data.w_vocab, data.c_vocab, h_params)
    if DEVICE.type == 'cuda':
        model.cuda(DEVICE)

    total_train_samples = len(data.train_iter)*BATCH_SIZE
    print("training examples: ", total_train_samples)
    current_time = time()
    #print(model)
    number_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.noise_model.infer_bn.parameters() if p.requires_grad)
    print("Inference parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.noise_model.gen_bn.parameters() if p.requires_grad)
    print("Generation parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.word_embeddings.parameters() if p.requires_grad)
    print("Word embedding parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.char_embeddings.parameters() if p.requires_grad)
    print("Char embedding parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    min_perp = 1e20
    wait_count = 0
    loss = torch.tensor(1e20)
    mean_loss = 0
    unk_index, pad_index = h_params.w_ignore_index, data.w_vocab.stoi['<pad>']
    while data.train_iter is not None:
        for i, training_batch in enumerate(data.train_iter):

            if model.step == h_params.anneal_kl[0]:
                model.optimizer = h_params.optimizer(model.parameters(), **h_params.optimizer_kwargs)
                print('Refreshed optimizer !')
                if model.step != 0 and not torch.isnan(loss):
                    model.save()
                    print('Saved model after it\'s pure reconstruction phase')

            # print([' '.join([data.vocab.itos[t] for t in text_i]) for text_i in training_batch.text[:2]])
            loss = model.opt_step(sample_dic(training_batch, flags.mode, unk_index, pad_index))

            mean_loss += loss
            if i % 30 == 0:
                mean_loss /= 30
                print("step:{}, loss:{}, seconds/step:{}".format(model.step, mean_loss, time()-current_time))
                mean_loss = 0
            if model.step% TEST_FREQ == TEST_FREQ-1:
                model.eval()
                try:
                    test_batch = next(val_iterator)
                except StopIteration:
                    print("Reinitialized test data iterator")
                    val_iterator = iter(data.val_iter)
                    test_batch = next(val_iterator)
                with torch.no_grad():
                    model(sample_dic(test_batch, flags.mode, unk_index, pad_index))
                model.dump_test_viz(complete=(model.step % COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1))
                model.train()
            current_time = time()
        data.reinit_iterator('valid')
        if model.step >= h_params.anneal_kl[0] and ((data.n_epochs % 3) == 0):
            model.eval()
            data.reinit_iterator('iw_valid')
            npp_ub, cpp_ub = model.get_perplexity(data.iw_val_iter, lambda batch: sample_dic(batch, flags.mode,
                                                                                             unk_index, pad_index))
            print("Noise Perplexity Upper Bound is {} at step {}".format(npp_ub, model.step))
            print("Clean Perplexity Upper Bound is {} at step {}".format(cpp_ub, model.step))
            data.reinit_iterator('valid')
            pp_ub = cpp_ub if flags.mode != "unsupervised" else npp_ub
            if pp_ub < min_perp:
                print('Saving The model ..')
                min_perp = pp_ub
                model.save()
                wait_count = 0
            else:
                wait_count += 1
            if flags.save_all:
                model.save()

            # if wait_count == flags.wait_epochs*2:
            #     break

            model.train()
        data.reinit_iterator('valid')
        data.reinit_iterator('train')
    print("Finished training")


def sample_dic(batch, mode, oov_index, unk_index):
    noise_y_orig = (1 - (batch.noise[..., 1:] == oov_index).long()).sum(-1).unsqueeze(-1).expand(batch.noise.shape)
    if mode == "unsupervised":
        # replacing <unk> tokens with pad tokens
        batch.noise[batch.noise == unk_index] = oov_index
        # shaping vectors as needed
        samples = {'noise': {'c': get_c(batch.c_noise), 'c_prev': get_c(batch.c_noise, prev=True),
                             'yorig': noise_y_orig, 'wid': batch.noise}}
        # for k, v in samples['noise'].items():
        #     print(k, v.shape)
    elif mode in ("distantly_supervised", "supervised", "semi_supervised"):
        # replacing <unk> tokens with pad tokens
        batch.noise[batch.noise == unk_index] = oov_index
        batch.noise[batch.clean == unk_index] = oov_index
        # shaping vectors as needed
        samples = {'clean': {'c': get_c(batch.c_clean), 'c_prev': get_c(batch.c_clean, prev=True),
                             'yorig': torch.ones(batch.c_clean[..., 1:].shape), 'wid': batch.clean},
                   'noise': {'c': get_c(batch.c_noise), 'c_prev': get_c(batch.c_noise, prev=True),
                             'yorig': noise_y_orig, 'wid':batch.noise}}
    else:
        raise NotImplementedError("Mode \"{}\" is not recognized as a training strategy.".format(mode))
    return samples


def get_c(c_samples, prev=False):
    if prev:
        return c_samples.view((*c_samples.shape[:-1], W_MAX_LEN+1, C_MAX_LEN+1))[..., :-1]
    else:
        return c_samples.view((*c_samples.shape[:-1], W_MAX_LEN+1, C_MAX_LEN+1))[..., 1:]


if __name__ == '__main__':
    main()


