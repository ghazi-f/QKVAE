# This file will implement the main training loop for a model
from time import time
import argparse

from torch import device
import torch
from torch import optim
import numpy as np

from data_prep import NLIGenData2, OntoGenData, HuggingYelp2
from disentanglement_transformer.models import DisentanglementTransformerVAE as Model
from disentanglement_transformer.h_params import DefaultTransformerHParams as HParams
from disentanglement_transformer.graphs import *
from components.criteria import *
parser = argparse.ArgumentParser()
from torch.nn import MultiheadAttention
# Training and Optimization
k, kz, klstm = 8, 16, 2
parser.add_argument("--test_name", default='unnamed', type=str)
parser.add_argument("--data", default='nli', choices=["nli", "ontonotes", "yelp"], type=str)
parser.add_argument("--max_len", default=17, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--grad_accu", default=1, type=int)
parser.add_argument("--n_epochs", default=10000, type=int)
parser.add_argument("--test_freq", default=32, type=int)
parser.add_argument("--complete_test_freq", default=160, type=int)
parser.add_argument("--generation_weight", default=1, type=float)
parser.add_argument("--device", default='cuda:0', choices=["cuda:0", "cuda:1", "cuda:2", "cpu"], type=str)
parser.add_argument("--embedding_dim", default=128, type=int)#################"
parser.add_argument("--pretrained_embeddings", default=False, type=bool)#################"
parser.add_argument("--z_size", default=96*kz, type=int)#################"
parser.add_argument("--z_emb_dim", default=192*k, type=int)#################"
parser.add_argument("--n_latents", default=[16, 16, 16], nargs='+', type=int)#################"
parser.add_argument("--text_rep_l", default=3, type=int)
parser.add_argument("--text_rep_h", default=192*k, type=int)
parser.add_argument("--encoder_h", default=192*k, type=int)#################"
parser.add_argument("--encoder_l", default=1, type=int)#################"
parser.add_argument("--decoder_h", default=192*k, type=int)
parser.add_argument("--decoder_l", default=1, type=int)#################"
parser.add_argument("--highway", default=False, type=bool)
parser.add_argument("--markovian", default=True, type=bool)
parser.add_argument('--minimal_enc', dest='minimal_enc', action='store_true')
parser.add_argument('--no-minimal_enc', dest='minimal_enc', action='store_false')
parser.set_defaults(minimal_enc=False)
parser.add_argument("--losses", default='VAE', choices=["VAE", "IWAE"], type=str)
parser.add_argument("--graph", default='Normal', choices=["Discrete", "IndepInfer", "Normal", "NormalConGen", "NormalSimplePrior",
                                                          "Normal2",  "NormalLSTM"], type=str)
parser.add_argument("--training_iw_samples", default=5, type=int)
parser.add_argument("--testing_iw_samples", default=4, type=int)
parser.add_argument("--test_prior_samples", default=10, type=int)
parser.add_argument("--anneal_kl0", default=3000, type=int)
parser.add_argument("--anneal_kl1", default=25000, type=int)
parser.add_argument("--grad_clip", default=5., type=float)
parser.add_argument("--kl_th", default=0/(768*k/2), type=float or None)
parser.add_argument("--max_elbo1", default=6.0, type=float)
parser.add_argument("--max_elbo2", default=4.0, type=float)
parser.add_argument("--max_elbo_choice", default=5, type=int)
parser.add_argument("--kl_beta", default=0.35, type=int)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--word_dropout", default=0.3, type=float)
parser.add_argument("--l2_reg", default=0, type=float)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--lr_reduction", default=4., type=float)
parser.add_argument("--wait_epochs", default=1, type=float)
parser.add_argument("--save_all", default=True, type=bool)

flags = parser.parse_args()

# Manual Settings, Deactivate before pushing
if True:
    flags.batch_size = 128
    flags.grad_accu = 1
    flags.max_len = 17
    flags.graph = "IndepInfer"
    flags.test_name = "nliLM/nlitest"
    flags.data = "nli"
    flags.n_latents = [4]

# torch.autograd.set_detect_anomaly(True)
GRAPH = {"Discrete": get_discrete_auto_regressive_graph,
         "IndepInfer": get_structured_auto_regressive_indep_graph,
         "Normal": get_structured_auto_regressive_graph,
         "NormalConGen": get_structured_auto_regressive_graphConGen,
         "Normal2": get_structured_auto_regressive_graph2,
         "NormalLSTM": get_lstm_graph,
         "NormalSimplePrior": get_structured_auto_regressive_simple_prior}[flags.graph]
if flags.graph == "NormalLSTM":
    flags.encoder_h = int(flags.encoder_h/k*klstm)
Data = {"nli": NLIGenData2, "ontonotes": OntoGenData, "yelp": HuggingYelp2}[flags.data]
MAX_LEN = flags.max_len
BATCH_SIZE = flags.batch_size
GRAD_ACCU = flags.grad_accu
N_EPOCHS = flags.n_epochs
TEST_FREQ = flags.test_freq
COMPLETE_TEST_FREQ = flags.complete_test_freq
DEVICE = device(flags.device)
# This prevents illegal memory access on multigpu machines (unresolved issue on torch's github)
if flags.device.startswith('cuda'):
    torch.cuda.set_device(int(flags.device[-1]))
LOSSES = {'IWAE': [IWLBo],
          'VAE': [ELBo]}[flags.losses]
#  LOSSES = [IWLBo]
ANNEAL_KL = [flags.anneal_kl0*flags.grad_accu, flags.anneal_kl1*flags.grad_accu]
LOSS_PARAMS = [1]
if flags.grad_accu > 1:
    LOSS_PARAMS = [w/flags.grad_accu for w in LOSS_PARAMS]


def main():
    data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE, pretrained=flags.pretrained_embeddings)
    h_params = HParams(len(data.vocab.itos), len(data.tags.itos), MAX_LEN, BATCH_SIZE, N_EPOCHS,
                       device=DEVICE, vocab_ignore_index=data.vocab.stoi['<pad>'], decoder_h=flags.decoder_h,
                       decoder_l=flags.decoder_l, encoder_h=flags.encoder_h, encoder_l=flags.encoder_l,
                       text_rep_h=flags.text_rep_h, text_rep_l=flags.text_rep_l,
                       test_name=flags.test_name, grad_accumulation_steps=GRAD_ACCU,
                       optimizer_kwargs={'lr': flags.lr, #'weight_decay': flags.l2_reg, 't0':100, 'lambd':0.},
                                         'weight_decay': flags.l2_reg, 'betas': (0.9, 0.99)},
                       is_weighted=[], graph_generator=GRAPH,
                       z_size=flags.z_size, embedding_dim=flags.embedding_dim, anneal_kl=ANNEAL_KL,
                       grad_clip=flags.grad_clip*flags.grad_accu, kl_th=flags.kl_th, highway=flags.highway,
                       losses=LOSSES, dropout=flags.dropout, training_iw_samples=flags.training_iw_samples,
                       testing_iw_samples=flags.testing_iw_samples, loss_params=LOSS_PARAMS, optimizer=optim.AdamW,
                       markovian=flags.markovian, word_dropout=flags.word_dropout, contiguous_lm=False,
                       test_prior_samples=flags.test_prior_samples, n_latents=flags.n_latents,
                       max_elbo=[flags.max_elbo_choice, flags.max_elbo1],  # max_elbo is paper's beta
                       z_emb_dim=flags.z_emb_dim, minimal_enc=flags.minimal_enc, kl_beta=flags.kl_beta)
    val_iterator = iter(data.val_iter)
    print("Words: ", len(data.vocab.itos), ", On device: ", DEVICE.type)
    print("Loss Type: ", flags.losses)
    model = Model(data.vocab, data.tags, h_params, wvs=data.wvs, dataset=flags.data)
    if DEVICE.type == 'cuda':
        model.cuda(DEVICE)

    total_unsupervised_train_samples = len(data.train_iter)*BATCH_SIZE
    total_unsupervised_val_samples = len(data.val_iter)*BATCH_SIZE
    print("Unsupervised training examples: ", total_unsupervised_train_samples)
    print("Unsupervised val examples: ", total_unsupervised_val_samples)
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
    min_perp = 1e20
    wait_count = 0
    loss = torch.tensor(1e20)
    mean_loss = 0
    stabilize_epochs = 0
    while data.train_iter is not None:
        for i, training_batch in enumerate(data.train_iter):
            if training_batch.text.shape[1] < 2: continue

            if model.step == h_params.anneal_kl[0]:
                model.optimizer = h_params.optimizer(model.parameters(), **h_params.optimizer_kwargs)
                print('Refreshed optimizer !')
                if model.step != 0 and not torch.isnan(loss):
                    model.save()
                    print('Saved model after it\'s pure reconstruction phase')

            # print([' '.join([data.vocab.itos[t] for t in text_i]) for text_i in training_batch.text[:2]])
            loss = model.opt_step({'x': training_batch.text[..., 1:], 'x_prev': training_batch.text[..., :-1]}) if flags.losses != 'S' else 0

            mean_loss += loss
            if i % 30 == 0:
                mean_loss /= 30
                print("step:{}, loss:{}, seconds/step:{}".format(model.step, mean_loss, time()-current_time))
                mean_loss = 0
            if int(model.step / (len(LOSSES))) % TEST_FREQ == TEST_FREQ-1:
                model.eval()
                try:
                    test_batch = limited_next(val_iterator)

                except StopIteration:
                    print("Reinitialized test data iterator")
                    val_iterator = iter(data.val_iter)
                    test_batch = limited_next(val_iterator)
                with torch.no_grad():
                    model({'x': test_batch.text[..., 1:], 'x_prev': test_batch.text[..., :-1]})
                model.dump_test_viz(complete=int(model.step / (len(LOSSES))) %
                                    COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)
                model.train()
            if model.step >= 7000:
                h_params.max_elbo = [flags.max_elbo_choice, flags.max_elbo2]
            current_time = time()
        data.reinit_iterator('valid')
        if model.step >= h_params.anneal_kl[0] and ((data.n_epochs % 3) == 0):
            model.eval()
            # pp_ub = model.get_perplexity(data.val_iter)
            if flags.data == "yelp":
                max_auc, auc_margin, max_auc_index  = model.get_sentiment_summaries(data.val_iter)
                print("max_auc: {}, auc_margin: {}, max_auc_index: {} ".format(max_auc, auc_margin, max_auc_index))
            # else:
            dis_diffs1, dis_diffs2, _, _ = model.get_disentanglement_summaries()
            print("disentanglement scores : {} and {}".format(dis_diffs1, dis_diffs2))

            # print("Perplexity Upper Bound is {} at step {}".format(pp_ub, model.step))
            data.reinit_iterator('valid')

            # if pp_ub < min_perp:
            #     print('Saving The model ..')
            #     min_perp = pp_ub
            #     model.save()
            #     wait_count = 0
            # else:
            #     wait_count += 1
            if flags.save_all:
                model.save()

            # if wait_count == flags.wait_epochs*2:
            #     break

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


