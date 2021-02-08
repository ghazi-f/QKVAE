# This file will implement the main training loop for a model
from time import time
import argparse
import os

from torch import device
import torch
from torch import optim
import numpy as np
from uuid import uuid4

from data_prep import PTBDaTA
from taln.models import GenerationModel as Model
from taln.h_params import DefaultSSSentenceClassificationHParams as HParams
from taln.graphs import *
from components.criteria import *
parser = argparse.ArgumentParser()

# Training and Optimization
parser.add_argument("--test_name", default='unnamed', type=str)
parser.add_argument("--result_csv", default='imdb.csv', type=str)
parser.add_argument("--max_len", default=72, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--grad_accu", default=1, type=int)
parser.add_argument("--n_epochs", default=10000, type=int)
parser.add_argument("--test_freq", default=20, type=int)
parser.add_argument("--complete_test_freq", default=160, type=int)
parser.add_argument("--generation_weight", default=1, type=float)
parser.add_argument("--device", default='cuda:0', choices=["cuda:0", "cuda:1", "cuda:2", "cpu"], type=str)
parser.add_argument("--embedding_dim", default=256, type=int)
parser.add_argument('--emb_batch_norm', dest='emb_batch_norm', action='store_true')
parser.add_argument('--no-emb_batch_norm', dest='emb_batch_norm', action='store_false')
parser.set_defaults(emb_batch_norm=False)
parser.add_argument("--divide_by", default=1, type=int)
parser.add_argument('--tied_embeddings', dest='tied_embeddings', action='store_true')
parser.add_argument('--no-tied_embeddings', dest='tied_embeddings', action='store_false')
parser.set_defaults(tied_embeddings=True)
parser.add_argument('--pretrained_embeddings', dest='pretrained_embeddings', action='store_true')
parser.add_argument('--no-pretrained_embeddings', dest='pretrained_embeddings', action='store_false')
parser.set_defaults(pretrained_embeddings=False)
parser.add_argument("--z_size", default=32, type=int)  # must be equal to encoder_h and decoder_h
parser.add_argument("--encoder_h", default=256, type=int)
parser.add_argument("--encoder_l", default=2, type=int)
parser.add_argument("--decoder_h", default=256, type=int)
parser.add_argument("--decoder_l", default=2, type=int)
parser.add_argument("--losses", default='VAE', choices=["VAE"], type=str)
parser.add_argument("--cycle_loss_w", default=0.0, type=float)
parser.add_argument("--training_iw_samples", default=5, type=int)
parser.add_argument("--testing_iw_samples", default=5, type=int)
parser.add_argument("--test_prior_samples", default=2, type=int)
parser.add_argument("--anneal_kl0", default=000, type=int)
parser.add_argument("--anneal_kl1", default=6000, type=int)
parser.add_argument("--grad_clip", default=0.5, type=float)
parser.add_argument("--kl_th", default=0.5/32, type=float or None)
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--word_dropout", default=0.2, type=float)
parser.add_argument("--l2_reg", default=0., type=float)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--opt_alg", default='adam', choices=["adam", "sgd", "nesterov"], type=str)
parser.add_argument("--beta1", default=0.9, type=float)#0.99, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--lr_decay", default=0.0, type=float)
parser.add_argument("--epsilon", default=1e-8, type=float)
parser.add_argument("--lr_reduction", default=4., type=float)
parser.add_argument("--wait_epochs", default=1, type=float)
parser.add_argument('--rm_save', dest='rm_save', action='store_true')
parser.add_argument('--no-rm_save', dest='rm_save', action='store_false')
parser.set_defaults(rm_save=False)

flags = parser.parse_args()
# Manual Settings, Deactivate before pushing
if True:
    flags.batch_size = 20
    flags.test_name = "tln\\githparamsth"
if True:
    flags.batch_size = 20
    flags.test_name = "tln\\manewthang2"
    flags.cycle_loss_w = 10.0
    flags.anneal_kl0 = 16000
    flags.anneal_kl1 = 24000
    flags.kl_th = 0.0
    flags.max_len = 64
    flags.z_size = 128
    flags.tied_embeddings = False
    flags.lr = 3e-4
    flags.grad_clip = 0.25

if flags.divide_by != 1:
    flags.embedding_dim = int(flags.embedding_dim/flags.divide_by)
    flags.z_size = int(flags.z_size/flags.divide_by)
    flags.encoder_h = int(flags.encoder_h/flags.divide_by)
    flags.decoder_h = int(flags.decoder_h/flags.divide_by)

if flags.pretrained_embeddings:
    flags.embedding_dim = 100
    #flags.tied_embeddings = True
    flags.decoder_h = flags.embedding_dim

# torch.autograd.set_detect_anomaly(True)
# flags.wait_epochs = int(flags.wait_epochs /flags.supervision_proportion )
Data = PTBDaTA
this_graph = get_generation_graph
MAX_LEN = flags.max_len
BATCH_SIZE = flags.batch_size
GRAD_ACCU = flags.grad_accu
N_EPOCHS = flags.n_epochs
TEST_FREQ = flags.test_freq
COMPLETE_TEST_FREQ = flags.complete_test_freq
DEVICE = device(flags.device)
OPT_ALG = {'adam': optim.AdamW, 'sgd': optim.SGD, 'nesterov': optim.SGD}[flags.opt_alg]
OPT_ARGS = {'adam': {'lr': flags.lr, 'weight_decay': flags.l2_reg, 'betas': (flags.beta1, flags.beta2),
                    'eps': flags.epsilon},
            'sgd': {'lr': flags.lr, 'momentum': flags.beta1, 'nesterov': False},
            'nesterov': {'lr': flags.lr, 'momentum': flags.beta1, 'nesterov': True}}[flags.opt_alg]

# This prevents illegal memory access on multigpu machines (unresolved issue on torch's github)
if flags.device.startswith('cuda'):
    torch.cuda.set_device(int(flags.device[-1]))
LOSSES = [ELBo]
ANNEAL_KL = [flags.anneal_kl0*flags.grad_accu, flags.anneal_kl1*flags.grad_accu] if flags.losses != 'S' else [0, 0]
LOSS_PARAMS = [1]
if flags.grad_accu > 1:
    LOSS_PARAMS = [w/flags.grad_accu for w in LOSS_PARAMS]


def main():
    global WARMED
    data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE, flags.pretrained_embeddings)
    h_params = HParams(len(data.vocab.itos), 0, MAX_LEN, BATCH_SIZE, N_EPOCHS,
                       device=DEVICE,  vocab_ignore_index=data.vocab.stoi['<pad>'], decoder_h=flags.decoder_h,
                       decoder_l=flags.decoder_l, encoder_h=flags.encoder_h, encoder_l=flags.encoder_l,
                       test_name=flags.test_name, grad_accumulation_steps=GRAD_ACCU,
                       optimizer_kwargs=OPT_ARGS,
                       is_weighted=[], graph_generator=this_graph, z_size=flags.z_size,
                       embedding_dim=flags.embedding_dim,
                       anneal_kl=ANNEAL_KL, grad_clip=flags.grad_clip,
                       kl_th=flags.kl_th, losses=LOSSES, dropout=flags.dropout,
                       training_iw_samples=flags.training_iw_samples, testing_iw_samples=flags.testing_iw_samples,
                       loss_params=LOSS_PARAMS, optimizer=OPT_ALG, word_dropout=flags.word_dropout, contiguous_lm=True,
                       tied_embeddings=flags.tied_embeddings, cycle_loss_w=flags.cycle_loss_w,
                       emb_batch_norm=flags.emb_batch_norm)
    val_iterator = iter(data.val_iter)
    print("Launching experiment ", flags.test_name)
    print("Words: ", len(data.vocab.itos), ", On device: ", DEVICE.type)
    print("Loss Type: ", flags.losses)
    model = Model(data.vocab, h_params, wvs=data.wvs)
    if flags.lr_decay > 0.0:
        lr_func = lambda epoch: 1 / (1.0 + epoch * flags.lr_decay)
        scheduler = optim.lr_scheduler.MultiplicativeLR(model.optimizer, lr_lambda=lr_func)
    else:
        scheduler = None
    if DEVICE.type == 'cuda':
        model.cuda(DEVICE)

    total_unsupervised_train_samples = len(data.train_iter)*BATCH_SIZE
    print("Unsupervised training examples: ", total_unsupervised_train_samples)
    current_time = time()
    step_times = []
    #print(model)
    number_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.infer_bn.parameters() if p.requires_grad)
    print("Inference parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.gen_bn.parameters() if p.requires_grad)
    print("Generation parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.word_embeddings.parameters() if p.requires_grad)
    print("Embedding parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    min_perp = 1e20
    wait_count = 0
    loss = torch.tensor(1e20)
    mean_loss = 0
    mean_cm = 0
    best_epoch = -1
    infer_prev, gen_prev = None, None

    while data.train_iter is not None:
        for i, training_batch in enumerate(data.train_iter):
            # if i < 1400: continue
            if training_batch.text.shape[1] < 2:
                print('Misshaped training sample')
                continue
            '''print([' '.join([data.vocab.itos[t] for t in text_i]) for text_i in training_batch.text[:2]])'''

            if model.step == h_params.anneal_kl[0]:
                model.optimizer = h_params.optimizer(model.parameters(), **h_params.optimizer_kwargs)
                print('Refreshed optimizer !')
                if model.step != 0 and not np.isnan(loss):
                    model.save()
                    print('Saved model after it\'s pure reconstruction phase')
            loss = model.opt_step({'x': training_batch.text[..., 1:], 'x_prev': training_batch.text[..., :-1]})

            mean_loss += loss
            mean_cm += model.cycle_metric or 0
            if i % 30 == 29:
                mean_loss /= 30
                mean_cm /= 30
                print("step:{}, loss:{}, cycle metric {}, seconds/step:{}".format(model.step, mean_loss, mean_cm,
                                                                                         np.mean(step_times)))
                step_times = []
                mean_loss = 0
                mean_cm = 0

            if int(model.step / (len(LOSSES))) % TEST_FREQ == TEST_FREQ-1:
                model.eval()
                try:
                    val_batch = limited_next(val_iterator)
                except StopIteration:
                    val_iterator = iter(data.val_iter)
                    print("Reinitialized test data iterator")
                    val_batch = limited_next(val_iterator)
                    infer_prev, gen_prev = None, None
                with torch.no_grad():
                    infer_prev, gen_prev = model({'x': val_batch.text[..., 1:], 'x_prev': val_batch.text[..., :-1]},
                                                 prev_states=(infer_prev, gen_prev))
                model.dump_test_viz(complete=model.step %
                                    COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)
                model.train()
            step_times.append(time()-current_time)
            current_time = time()

        data.reinit_iterator('valid')
        if model.step >= h_params.anneal_kl[0]:
            model.eval()
            pp_ub = model.get_perplexity(data.val_iter)
            print("Perplexity Upper Bound is {} at step {}".format(pp_ub, model.step))
            data.reinit_iterator('valid')
            # model.reduce_lr(flags.lr_reduction)
            # print('Learning rate reduced to ', [gr['lr'] for gr in model.optimizer.param_groups])
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
            if wait_count >= flags.wait_epochs *2 and model.step > 50:
                break
            model.train()
        data.reinit_iterator('valid')
        data.reinit_iterator('train')

    print("Finished training, starting final evaluation :")
    # Ended Training
    # Reloading best parameters
    model.load()
    # Getting final test numbers
    model.eval()

    pp_ub = model.get_perplexity(data.test_iter).item()
    print("Final test perplexity is: {}".format(pp_ub))
    if flags.rm_save:
        os.remove(h_params.save_path)

    if not os.path.exists(flags.result_csv):
        with open(flags.result_csv, 'w') as f:
            f.write(', '.join(['test_name', 'loss_type', 'generation_weight', 'pp_ub', 'best_epoch', 'embedding_dim',
                               'z_size', 'encoder_h', 'encoder_l', 'decoder_h', 'decoder_l', 'training_iw_samples',
                               'is_tied', 'pretrained', 'opt_alg', 'beta1', 'beta2', 'lr', 'batch_size', 'dropout',
                               'emb_batch_norm', 'epsilon'
                               ]) + '\n')

    with open(flags.result_csv, 'a') as f:
        f.write(', '.join([flags.test_name, flags.losses, str(flags.generation_weight), str(pp_ub), str(best_epoch),
                           str(flags.embedding_dim), str(flags.z_size), str(flags.encoder_h), str(flags.encoder_l),
                           str(flags.decoder_h), str(flags.decoder_l), str(flags.training_iw_samples),
                           str(flags.tied_embeddings), str(flags.pretrained_embeddings), flags.opt_alg,
                           str(flags.beta1), str(flags.beta2), str(flags.lr), str(flags.batch_size),
                           str(flags.dropout), str(flags.emb_batch_norm), str(flags.epsilon)
                           ])+'\n')


def limited_next(iterator):
    batch = next(iterator)
    if len(batch.text[0]) > MAX_LEN:
        batch.text = batch.text[:, :MAX_LEN]
        batch.label = batch.label[:, :MAX_LEN-1]
    return batch


if __name__ == '__main__':
    main()

