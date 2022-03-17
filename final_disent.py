# This file will implement the main training loop for a model
from time import time
import argparse
import os

from torch import device
import torch
from torch import optim
from transformers import Adafactor
import numpy as np
from allennlp.training.learning_rate_schedulers import PolynomialDecay

from disentanglement_final.data_prep import NLIGenData2, OntoGenData, HuggingYelp2, ParaNMTCuratedData, BARTYelp, \
    BARTParaNMT, BARTNLI, BARTNewsCategory, BARTFrSbt, BARTWiki, BARTBookCorpus
from disentanglement_final.models import DisentanglementTransformerVAE, StructuredDisentanglementVAE
from disentanglement_final.h_params import DefaultTransformerHParams as HParams
from disentanglement_final.graphs import *
from components.criteria import *
parser = argparse.ArgumentParser()
from torch.nn import MultiheadAttention
# Training and Optimization
k, kz, klstm = 2, 4, 2
parser.add_argument("--test_name", default='unnamed', type=str)
parser.add_argument("--data", default='nli', choices=["nli", "ontonotes", "yelp", 'paranmt', 'news', 'fr_sbt', 'wiki',
                                                      'bc'], type=str)
parser.add_argument("--csv_out", default='disentqkv3.csv', type=str)
parser.add_argument("--max_len", default=17, type=int)
parser.add_argument("--init_len", default=None, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--grad_accu", default=1, type=int)
parser.add_argument("--n_epochs", default=20, type=int)
parser.add_argument("--test_freq", default=32, type=int)
parser.add_argument("--complete_test_freq", default=160, type=int)
parser.add_argument("--generation_weight", default=1, type=float)
parser.add_argument("--device", default='cuda:0', choices=["cuda:0", "cuda:1", "cuda:2", "cpu"], type=str)
parser.add_argument("--embedding_dim", default=128, type=int)#################"
parser.add_argument("--pretrained_embeddings", default=False, type=bool)#################"
parser.add_argument("--z_size", default=96*kz, type=int)#################"
parser.add_argument("--z_emb_dim", default=192*k, type=int)#################"
parser.add_argument("--n_keys", default=4, type=int)#################"
parser.add_argument("--n_latents", default=[4], nargs='+', type=int)#################"
parser.add_argument("--n_heads", default=12, type=int)#################"
parser.add_argument("--n_aux_mem", default=10, type=int)#################"
parser.add_argument("--text_rep_l", default=3, type=int)
parser.add_argument("--text_rep_h", default=192*k, type=int)
parser.add_argument("--encoder_h", default=192*k, type=int)#################"
parser.add_argument("--encoder_l", default=2, type=int)#################"
parser.add_argument("--decoder_h", default=int(192*k), type=int)################
parser.add_argument("--decoder_l", default=2, type=int)#################"
parser.add_argument("--highway", default=False, type=bool)
parser.add_argument("--markovian", default=True, type=bool)
parser.add_argument('--minimal_enc', dest='minimal_enc', action='store_true')
parser.add_argument('--no-minimal_enc', dest='minimal_enc', action='store_false')
parser.set_defaults(minimal_enc=False)
parser.add_argument('--use_bart', dest='use_bart', action='store_true')
parser.add_argument('--no-use_bart', dest='use_bart', action='store_false')
parser.set_defaults(use_bart=False)
parser.add_argument('--layer_wise_qkv', dest='layer_wise_qkv', action='store_true')
parser.add_argument('--no-layer_wise_qkv', dest='layer_wise_qkv', action='store_false')
parser.set_defaults(layer_wise_qkv=False)
parser.add_argument('--tr_enc_in_dec', dest='tr_enc_in_dec', action='store_true')
parser.add_argument('--no-tr_enc_in_dec', dest='tr_enc_in_dec', action='store_false')
parser.set_defaults(tr_enc_in_dec=False)
parser.add_argument("--losses", default='VAE', choices=["VAE", "IWAE", "LagVAE"], type=str)
parser.add_argument("--graph", default='Normal', choices=["Vanilla", "IndepInfer", "QKV", "SQKV", "HQKV", "HQKVDiscZs",
                                                          "NQKV"],
                    type=str)
parser.add_argument("--training_iw_samples", default=1, type=int)
parser.add_argument("--testing_iw_samples", default=5, type=int)
parser.add_argument("--test_prior_samples", default=10, type=int)
parser.add_argument("--anneal_kl0", default=3000, type=int)
parser.add_argument("--anneal_kl1", default=6000, type=int)
parser.add_argument("--zs_anneal_kl0", default=7000, type=int)
parser.add_argument("--zs_anneal_kl1", default=10000, type=int)
parser.add_argument("--zg_anneal_kl0", default=7000, type=int)
parser.add_argument("--zg_anneal_kl1", default=10000, type=int)
parser.add_argument("--anneal_kl_type", default="linear", choices=["linear", "sigmoid"], type=str)
parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"], type=str)
parser.add_argument("--grad_clip", default=5., type=float)
parser.add_argument("--kl_th", default=0., type=float or None)
parser.add_argument("--max_elbo1", default=6.0, type=float)
parser.add_argument("--max_elbo2", default=4.0, type=float)
parser.add_argument("--max_elbo_choice", default=6, type=int)
parser.add_argument("--kl_beta", default=0.3, type=float)
parser.add_argument("--kl_beta_zs", default=0.1, type=float)
parser.add_argument("--kl_beta_zg", default=0.1, type=float)
parser.add_argument("--lv_kl_coeff", default=0.0, type=float)
parser.add_argument("--sem_coeff", default=0.0, type=float)
parser.add_argument("--dropout", default=0.3, type=float)
parser.add_argument("--word_dropout", default=0.4, type=float)
parser.add_argument("--l2_reg", default=0, type=float)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--lr_sched", default=0., type=float)
parser.add_argument("--lr_reduction", default=4., type=float)
parser.add_argument("--wait_epochs", default=1, type=float)
parser.add_argument("--save_all", default=True, type=bool)

flags = parser.parse_args()

# Manual Settings, Deactivate before pushing
if False:
    # flags.optimizer="sgd"
    flags.use_bart = True
    flags.layer_wise_qkv = True
    flags.tr_enc_in_dec = True
    # flags.lr_sched = 0.00003
    flags.batch_size = 8
    flags.grad_accu = 1
    flags.max_len = 4
    flags.test_name = "nliLM/TestBart"
    # flags.lv_kl_coeff = 1.0
    flags.data = "bc"#"fr_sbt"
    flags.sem_coeff = 1.0
    flags.n_latents = [4]
    flags.n_keys = 16
    flags.graph = "NQKV"  # "Vanilla"
    flags.z_size = 192
    flags.losses = "VAE"
    flags.kl_beta = 0.4
    flags.kl_beta_zg = 0.1
    flags.kl_beta_zs = 0.01
    # flags.encoder_h = 768
    # flags.decoder_h = 768
    # flags.anneal_kl0, flags.anneal_kl1 = 4000, 500
    # flags.zs_anneal_kl0, flags.zs_anneal_kl1 = 6000, 500
    # flags.zg_anneal_kl0, flags.zg_anneal_kl1 = 6000, 500
    flags.word_dropout = 0.4
    # flags.anneal_kl_type = "sigmoid"
    # flags.encoder_l = 4
    # flags.decoder_l = 4

    # flags.anneal_kl0 = 0
    flags.max_elbo_choice = 6
    # flags.z_size = 16
    # flags.encoder_h = 256
    # flags.decoder_h = 256

if flags.use_bart:
    # flags.z_size = 768
    flags.decoder_h = 768
    flags.encoder_h = 768
    flags.embedding_dim = 768

if flags.anneal_kl_type == "sigmoid" and flags.anneal_kl0 < flags.anneal_kl1:
    flags.anneal_kl0, flags.anneal_kl1 = 2000, 500
    flags.zs_anneal_kl0, flags.zs_anneal_kl1 = 4000, 500
    flags.zg_anneal_kl0, flags.zg_anneal_kl1 = 4000, 500


if flags.use_bart and flags.optimizer == "adam": flags.optimizer = "adafactor"
OPTIMIZER = {'sgd': optim.SGD, 'adam': optim.Adam, "adafactor": Adafactor}[flags.optimizer]
OPT_KWARGS = {'sgd': {'lr': flags.lr, 'weight_decay': flags.l2_reg},  # 't0':100, 'lambd':0.},
              'adam': {'lr': flags.lr, 'weight_decay': flags.l2_reg, 'betas': (0.9, 0.99)},
              'adafactor': {'lr': flags.lr, 'relative_step': False,
                            'weight_decay': flags.l2_reg}}[flags.optimizer]

# torch.autograd.set_detect_anomaly(True)
GRAPH = {"Vanilla": get_vanilla_graph,
         "IndepInfer": get_BARTADVAE if flags.use_bart else get_structured_auto_regressive_indep_graph,
         "QKV": get_qkv_graphBART if flags.use_bart else get_qkv_graph2,
         "SQKV": get_min_struct_qkv_graphBART if flags.use_bart else None,
         "HQKV": get_hqkv_graphBART if flags.use_bart else get_hqkv_graph,
         "HQKVDiscZs": get_hqkv_graph_discrete_zsBART if flags.use_bart else get_hqkv_graph_discrete_zs,
         "NQKV":get_qkvNext}[flags.graph]
if flags.graph == "NormalLSTM":
    flags.encoder_h = int(flags.encoder_h/k*klstm)
if flags.graph == "Vanilla":
    flags.n_latents = [flags.z_size]
if flags.losses == "LagVAE":
    flags.anneal_kl0, flags.zs_anneal_kl0, flags.zg_anneal_kl0 = 0, 0, 0
    flags.anneal_kl1, flags.zs_anneal_kl1, flags.zg_anneal_kl1 = 0, 0, 0
    # flags.kl_beta, flags.kl_beta_zs, flags.kl_beta_zg = 1.0, 1.0, 1.0

if flags.data in ('news', 'fr_sbt', 'wiki'): assert flags.use_bart
Data = {"nli": BARTNLI if flags.use_bart else NLIGenData2, "ontonotes": OntoGenData,
        "yelp": BARTYelp if flags.use_bart else HuggingYelp2,
        "paranmt": BARTParaNMT if flags.use_bart else ParaNMTCuratedData,
        "news": BARTNewsCategory, "wiki": BARTWiki, "bc": BARTBookCorpus,
        'fr_sbt': BARTFrSbt}[flags.data]
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
          'VAE': [ELBo],
          'LagVAE': [ELBo]}[flags.losses]

ANNEAL_KL = [flags.anneal_kl0*flags.grad_accu, flags.anneal_kl1*flags.grad_accu]
ZS_ANNEAL_KL = [flags.zs_anneal_kl0*flags.grad_accu, flags.zs_anneal_kl1*flags.grad_accu]
ZG_ANNEAL_KL = [flags.zg_anneal_kl0*flags.grad_accu, flags.zg_anneal_kl1*flags.grad_accu]
LOSS_PARAMS = [1]
if flags.grad_accu > 1:
    LOSS_PARAMS = [w/flags.grad_accu for w in LOSS_PARAMS]


def main():
    data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE, pretrained=flags.pretrained_embeddings)
    h_params = HParams(len(data.vocab.itos), len(data.tags.itos) if (flags.data == 'yelp' and not flags.use_bart)
                       else None, MAX_LEN, BATCH_SIZE, N_EPOCHS, layer_wise_qkv=flags.layer_wise_qkv,
                       device=DEVICE, vocab_ignore_index=data.vocab.stoi['<pad>'], decoder_h=flags.decoder_h,
                       decoder_l=flags.decoder_l, encoder_h=flags.encoder_h, encoder_l=flags.encoder_l,
                       text_rep_h=flags.text_rep_h, text_rep_l=flags.text_rep_l, n_heads=flags.n_heads,
                       n_aux_mem=flags.n_aux_mem, test_name=flags.test_name, grad_accumulation_steps=GRAD_ACCU,
                       optimizer_kwargs=OPT_KWARGS, tr_enc_in_dec=flags.tr_enc_in_dec,
                       is_weighted=[], graph_generator=GRAPH, z_size=flags.z_size, embedding_dim=flags.embedding_dim,
                       anneal_kl=ANNEAL_KL, zs_anneal_kl=ZS_ANNEAL_KL, zg_anneal_kl=ZG_ANNEAL_KL,
                       grad_clip=flags.grad_clip*flags.grad_accu, kl_th=flags.kl_th, highway=flags.highway,
                       losses=LOSSES, dropout=flags.dropout, training_iw_samples=flags.training_iw_samples,
                       testing_iw_samples=flags.testing_iw_samples, loss_params=LOSS_PARAMS, optimizer=OPTIMIZER,
                       markovian=flags.markovian, word_dropout=flags.word_dropout, contiguous_lm=False,
                       test_prior_samples=flags.test_prior_samples, n_latents=flags.n_latents, n_keys=flags.n_keys,
                       max_elbo=[flags.max_elbo_choice, flags.max_elbo1],  lv_kl_coeff=flags.lv_kl_coeff,sem_coeff=flags.sem_coeff,
                       z_emb_dim=flags.z_emb_dim, minimal_enc=flags.minimal_enc, kl_beta=flags.kl_beta,
                       kl_beta_zs=flags.kl_beta_zs, kl_beta_zg=flags.kl_beta_zg, anneal_kl_type=flags.anneal_kl_type,
                       fr=flags.data == 'fr_sbt')
    val_iterator = iter(data.val_iter)
    print("Words: ", len(data.vocab.itos), ", On device: ", DEVICE.type, flush=True)
    print("Loss Type: ", flags.losses)
    if flags.graph == "NQKV":
        model = StructuredDisentanglementVAE(data.vocab, data.tags, h_params, wvs=data.wvs, dataset=data)
    else:
        model = DisentanglementTransformerVAE(data.vocab, data.tags, h_params, wvs=data.wvs, dataset=data)
    if DEVICE.type == 'cuda':
        model.cuda(DEVICE)

    # Redefining examples lengths:
    if flags.init_len is not None:
        data.redefine_max_len(flags.init_len)
        h_params.max_len = flags.init_len

    if flags.lr_sched > 0:
        decay = PolynomialDecay(optimizer=model.optimizer, num_epochs=1, num_steps_per_epoch=100000, power=2.0,
                                warmup_steps=500, end_learning_rate=flags.lr_sched)# typically 3e-5

    total_unsupervised_train_samples = len(data.train_iter)*BATCH_SIZE
    total_unsupervised_val_samples = len(data.val_iter)*(BATCH_SIZE/data.divide_bs)
    print("Unsupervised training examples: ", total_unsupervised_train_samples)
    print("Unsupervised val examples: ", total_unsupervised_val_samples)
    number_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.infer_bn.parameters() if p.requires_grad)
    print("Inference parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.gen_bn.parameters() if p.requires_grad)
    print("Generation parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.word_embeddings.parameters() if p.requires_grad)
    print("Embedding parameters: ", "{0:05.2f} M".format(number_parameters/1e6))

    current_time = time()
    loss = torch.tensor(1e20)
    mean_loss = 0
    model.beam_size = 4
    prev_mi = 0
    # model.eval()
    # orig_mod_bleu, para_mod_bleu, rec_bleu = model.get_paraphrase_bleu(data.val_iter, beam_size=5)
    # print(orig_mod_bleu, para_mod_bleu, rec_bleu)
    # model.step = 8000
    # model.get_disentanglement_summaries2(data.val_iter, 200)
    # print(model.get_syn_disent_encoder(split="valid"))
    # dev_kl, dev_kl_std, dev_rec, val_mi = model.collect_stats(data.val_iter)
    # pp_ub = model.get_perplexity(data.val_iter)
    # print(model.get_swap_tma(n_samples=200))
    while data.train_iter is not None:
        # ============================= TRAINING LOOP ==================================================================
        for i, training_batch in enumerate(data.train_iter):
            # print("Training iter ", i, flush=True)
            if training_batch.text.shape[1] < 2: continue

            if model.step == h_params.anneal_kl[0]:
                model.optimizer = h_params.optimizer(model.parameters(), **h_params.optimizer_kwargs)
                print('Refreshed optimizer !')
                if model.step != 0 and not torch.isnan(loss):
                    model.save()
                    print('Saved model after it\'s pure reconstruction phase')

            # print([' '.join([data.vocab.itos[t] for t in text_i]) for text_i in training_batch.text[:2]])
            train_inp = {'x': training_batch.text[..., 1:], 'x_prev': training_batch.text[..., :-1]}
            if flags.graph == "NQKV":
                train_inp["x+"] = training_batch.next[..., 1:]
                train_inp["x_prev+"] = training_batch.next[..., :-1]
            loss = model.opt_step(train_inp)
            if flags.lr_sched > 0:
                decay.step_batch()

                # ---------------------- In-training metric calculations ---------------------------------------------------
            mean_loss += loss
            if i % 30 == 0:
                mean_loss /= 30
                print("step:{}, loss:{}, seconds/step:{}".format(model.step, mean_loss, time()-current_time), flush=True)
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
                    test_inp = {'x': test_batch.text[..., 1:], 'x_prev': test_batch.text[..., :-1]}
                    if flags.graph == "NQKV":
                        test_inp["x+"] = test_batch.next[..., 1:]
                        test_inp["x_prev+"] = test_batch.next[..., :-1]
                    is_complete = (int(model.step / (len(LOSSES))) % COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)\
                                   and (model.step > flags.anneal_kl0) and (model.step > flags.anneal_kl1)
                    model(test_inp, sem_loss=not is_complete)
                    for loss in model.losses:
                        print(type(loss), ":")
                        print(loss._prepared_metrics, flush=True)
                    model.dump_test_viz(complete=is_complete)
            # ----------------------------------------------------------------------------------------------------------
                model.train()
            current_time = time()
        # ============================= EPOCH-WISE EVAL ================================================================
        data.reinit_iterator('valid')
        if model.step >= h_params.anneal_kl[0]:  # and ((data.n_epochs % 3) == 0):
            model.eval()
            pp_ub = model.get_perplexity(data.val_iter)
            print("perplexity is {} ".format(pp_ub))
            # if flags.data == "yelp":
            #     max_auc, auc_margin, max_auc_index  = model.get_sentiment_summaries(data.val_iter)
            #     print("max_auc: {}, auc_margin: {}, max_auc_index: {} ".format(max_auc, auc_margin, max_auc_index))
            # if flags.data == "paranmt":
            #     orig_mod_bleu, para_mod_bleu, rec_bleu = model.get_paraphrase_bleu(data.val_iter)
            #     print("orig_mod_bleu: {}, para_mod_bleu: {}, rec_bleu: {} ".format(orig_mod_bleu, para_mod_bleu, rec_bleu))

            print("=========== Old disentanglement scores ========================")
            val_dec_lab_wise_disent, val_enc_lab_wise_disent, val_decoder_Ndisent_vars, val_encoder_Ndisent_vars\
                = model.get_disentanglement_summaries2(data.val_iter, 200)
            print("Encoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_enc_lab_wise_disent,
                                                                           sum(val_enc_lab_wise_disent.values()),
                                                                                      val_encoder_Ndisent_vars))
            print("Decoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_dec_lab_wise_disent,
                                                                           sum(val_dec_lab_wise_disent.values()),
                                                                                      val_decoder_Ndisent_vars))

            print("=========== New syntax disentanglement scores ========================")
            val_encoder_syn_disent_scores = model.get_syn_disent_encoder(split="valid")
            if flags.graph not in ("Vanilla", "IndepInfer"):
                decoder_syn_disent_scores = model.get_swap_tma(n_samples=200)
            else:
                decoder_syn_disent_scores = {"tma2": {"zs": 0, "zc": 0, "copy": 0},
                                             "tma3": {"zs": 0, "zc": 0, "copy": 0},
                                             "bleu": {"zs": 0, "zc": 0, "copy": 0}}
            print("Encoder Syntax Disentanglement Scores: ", val_encoder_syn_disent_scores)
            print("Decoder Syntax Disentanglement Scores: ", decoder_syn_disent_scores)

            # print("Perplexity Upper Bound is {} at step {}".format(pp_ub, model.step))
            data.reinit_iterator('valid')

            dev_kl, dev_kl_std, dev_rec, val_mi = model.collect_stats(data.val_iter)
            data.reinit_iterator('valid')
            if val_mi < prev_mi and flags.losses == "LagVAE":
                print("Stopped aggressive training phase")
                model.aggressive = False
            prev_mi = val_mi

            if flags.save_all:
                print('Saving The model ..')
                model.save()

            model.train()
        data.reinit_iterator('valid')
        data.reinit_iterator('train')
    print("================= Finished training : Getting Scores on test set ============")
    model.eval()

    print("================= Old Disentanglement Scores ============")
    val_dec_lab_wise_disent, val_enc_lab_wise_disent, val_decoder_Ndisent_vars, val_encoder_Ndisent_vars\
        = model.get_disentanglement_summaries2(data.val_iter)
    print("Encoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_enc_lab_wise_disent,
                                                                              sum(val_enc_lab_wise_disent.values()),
                                                                              val_encoder_Ndisent_vars))
    print("Decoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_dec_lab_wise_disent,
                                                                              sum(val_dec_lab_wise_disent.values()),
                                                                              val_decoder_Ndisent_vars))
    test_dec_lab_wise_disent, test_enc_lab_wise_disent, test_decoder_Ndisent_vars, test_encoder_Ndisent_vars\
        = model.get_disentanglement_summaries2(data.test_iter)
    data.reinit_iterator('test')
    print("Encoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(test_enc_lab_wise_disent,
                                                                              sum(test_enc_lab_wise_disent.values()),
                                                                              test_encoder_Ndisent_vars))
    print("Decoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(test_dec_lab_wise_disent,
                                                                              sum(test_dec_lab_wise_disent.values()),
                                                                              test_decoder_Ndisent_vars))
    print("=========== New syntax disentanglement scores ========================")
    val_encoder_syn_disent_scores = model.get_syn_disent_encoder(split="valid")
    test_encoder_syn_disent_scores = model.get_syn_disent_encoder(split="test")
    if flags.graph not in ("Vanilla", "IndepInfer"):
        decoder_syn_disent_scores = model.get_swap_tma()
    else:
        decoder_syn_disent_scores = {"tma2": {"zs": 0, "zc": 0, "copy": 0}, "tma3": {"zs": 0, "zc": 0, "copy": 0},
                                     "bleu": {"zs": 0, "zc": 0, "copy": 0}}
    print("Encoder Syntax Disentanglement Scores: ", val_encoder_syn_disent_scores)
    print("Encoder Syntax Disentanglement Scores: ", test_encoder_syn_disent_scores)
    print("Decoder Syntax Disentanglement Scores: ", decoder_syn_disent_scores)

    print("=========== General VAE Language Modeling Metrics ========================")
    pp_ub = model.get_perplexity(data.val_iter)
    test_pp_ub = model.get_perplexity(data.test_iter)
    print("Perplexity: {}".format(test_pp_ub))
    dev_kl, dev_kl_std, dev_rec, val_mi = model.collect_stats(data.val_iter)
    test_kl, test_kl_std, test_rec, test_mi = model.collect_stats(data.test_iter)
    relations = ['subj', 'verb', 'dobj', 'pobj']
    temps = ['syntemp', 'lextemp']
    enc_tasks, dec_tasks = ["template", "paraphrase"], ["tma2", "tma3", "bleu"]
    enc_vars, dec_vars = ["zs", "zc", 'zp'], ["zs", "zc", 'zp', "copy"]
    if not os.path.exists(flags.csv_out):
        with open(flags.csv_out, 'w') as f:
            label_line = ['name', 'net_size', 'z_size', 'graph', 'data', 'kl_beta', 'n_latents', 'n_keys',
                               'dev_kl', 'dev_kl_std', 'dev_ppl', 'dev_tot_dec_disent',
                              'dev_tot_en_disent', 'dev_dec_disent_subj', 'dev_dec_disent_verb', 'dev_dec_disent_dobj',
                              'dev_dec_disent_syntemp', 'dev_dec_disent_lextemp',
                              'dev_dec_disent_pobj', 'dev_enc_disent_subj', 'dev_enc_disent_verb', 'dev_enc_disent_dobj',
                              'dev_enc_disent_pobj', 'dev_rec_error', 'dev_decoder_Ndisent_vars', 'dev_encoder_Ndisent_vars',
                              'test_kl', 'test_kl_std', 'test_ppl', 'test_tot_dec_disent',
                              'test_tot_en_disent', 'test_dec_disent_subj', 'test_dec_disent_verb', 'test_dec_disent_dobj',
                              'test_dec_disent_syntemp', 'test_dec_disent_lextemp',
                              'test_dec_disent_pobj', 'test_enc_disent_subj', 'test_enc_disent_verb', 'test_enc_disent_dobj',
                              'test_enc_disent_pobj', 'test_rec_error', 'test_decoder_Ndisent_vars', 'test_encoder_Ndisent_vars',
                              'dev_mi', 'test_mi']
            for t in enc_tasks:
                for v in enc_vars:
                    label_line.append("_".join([v, t, "dev", "score"]))
                    label_line.append("_".join([v, t, "test", "score"]))
            for t in dec_tasks:
                for v in dec_vars:
                    label_line.append("_".join([v, t, "score"]))
            f.write('\t'.join(label_line)+'\n')
    with open(flags.csv_out, 'a') as f:
        value_line = [flags.test_name, str(flags.encoder_h), str(flags.z_size), str(flags.graph), str(flags.data),
                           str(flags.kl_beta), str(flags.n_latents), str(flags.n_keys),
                           str(dev_kl), str(dev_kl_std), str(pp_ub), str(sum(val_dec_lab_wise_disent.values())),
                           str(sum(val_enc_lab_wise_disent.values())),
                           *[str(val_dec_lab_wise_disent[k]) for k in relations+temps],
                           *[str(val_enc_lab_wise_disent[k]) for k in relations], str(dev_rec),
                           str(val_decoder_Ndisent_vars), str(val_encoder_Ndisent_vars),
                           str(test_kl), str(test_kl_std), str(test_pp_ub), str(sum(test_dec_lab_wise_disent.values())),
                           str(sum(test_enc_lab_wise_disent.values())),
                           *[str(test_dec_lab_wise_disent[k]) for k in relations+temps],
                           *[str(test_enc_lab_wise_disent[k]) for k in relations], str(test_rec),
                           str(test_decoder_Ndisent_vars), str(test_encoder_Ndisent_vars), str(val_mi), str(test_mi)
                      ]
        for t in enc_tasks:
            for v in enc_vars:
                value_line.append(str(val_encoder_syn_disent_scores[t][v]))
                value_line.append(str(test_encoder_syn_disent_scores[t][v]))
        for t in dec_tasks:
            for v in dec_vars:
                value_line.append(str(decoder_syn_disent_scores[t][v]))
        f.write('\t'.join(value_line)+'\n')

    print("Finished training !")


def limited_next(iterator):
    batch = next(iterator)
    if len(batch.text[0]) > MAX_LEN:
        batch.text = batch.text[:, :MAX_LEN]
        batch.label = batch.label[:, :MAX_LEN-2]
    return batch


if __name__ == '__main__':
    main()


