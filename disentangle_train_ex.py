# This file will implement the main training loop for a model
from time import time
import argparse
import os

from torch import device
import torch
from torch import optim
import numpy as np

from disentanglement_transformer_extended.data_prep import NLIGenData2, OntoGenData, HuggingYelp2, HuggingYelpReg,\
    GermanNLIGenData2, SupYelpData, SupNLIData
from disentanglement_transformer_extended.models import DisentanglementTransformerVAE, LaggingDisentanglementTransformerVAE
from disentanglement_transformer_extended.h_params import DefaultTransformerHParams as HParams
from disentanglement_transformer_extended.graphs import *
from components.criteria import *
parser = argparse.ArgumentParser()
from torch.nn import MultiheadAttention
# Training and Optimization
k, kz, klstm = 1, 8, 2
parser.add_argument("--test_name", default='unnamed', type=str)
parser.add_argument("--data", default='nli', choices=["nli", "ontonotes", "yelp", "yelp_reg", "de_nli", "sup_yelp",
                                                      "sup_nli"], type=str)
parser.add_argument("--csv_out", default='disentICLRDE.csv', type=str)
parser.add_argument("--max_len", default=17, type=int)
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
parser.add_argument("--n_latents", default=[16, 16, 16], nargs='+', type=int)#################"
parser.add_argument("--text_rep_l", default=3, type=int)
parser.add_argument("--text_rep_h", default=192*k, type=int)
parser.add_argument("--encoder_h", default=768*k, type=int)#################"
parser.add_argument("--encoder_l", default=2, type=int)#################"
parser.add_argument("--decoder_h", default=768*k, type=int)
parser.add_argument("--decoder_l", default=2, type=int)#################"
parser.add_argument("--n_heads", default=4, type=int)#################"
parser.add_argument("--highway", default=False, type=bool)
parser.add_argument("--markovian", default=True, type=bool)
parser.add_argument('--minimal_enc', dest='minimal_enc', action='store_true')
parser.add_argument('--no-minimal_enc', dest='minimal_enc', action='store_false')
parser.set_defaults(minimal_enc=False)
parser.add_argument('--no_sa', dest='no_sa', action='store_true')
parser.add_argument('--no-no_sa', dest='no_sa', action='store_false')
parser.set_defaults(no_sa=True)
parser.add_argument('--tr_enc_in_dec', dest='tr_enc_in_dec', action='store_true')
parser.add_argument('--no-tr_enc_in_dec', dest='tr_enc_in_dec', action='store_false')
parser.set_defaults(tr_enc_in_dec=False)
parser.add_argument("--losses", default='VAE', choices=["VAE", "IWAE" "LagVAE"], type=str)
parser.add_argument("--graph", default='Normal', choices=["Vanilla", "Discrete", "IndepInfer", "Normal", "NormalConGen",
                                                          "NormalSimplePrior", "Normal2",  "NormalLSTM", "VanillaTr"],
                    type=str)
parser.add_argument("--training_iw_samples", default=1, type=int)
parser.add_argument("--testing_iw_samples", default=20, type=int)
parser.add_argument("--test_prior_samples", default=10, type=int)
parser.add_argument("--anneal_kl0", default=3000, type=int)
parser.add_argument("--anneal_kl1", default=6000, type=int)
parser.add_argument("--grad_clip", default=5., type=float)
parser.add_argument("--kl_th", default=0/(768*k/2), type=float or None)
parser.add_argument("--max_elbo1", default=6.0, type=float)
parser.add_argument("--max_elbo2", default=4.0, type=float)
parser.add_argument("--max_elbo_choice", default=10, type=int)
parser.add_argument("--kl_beta", default=0.4, type=float)
parser.add_argument("--lv_kl_coeff", default=0.0, type=float)
parser.add_argument("--sup_coeff", default=0.0, type=float)
parser.add_argument("--dec_sup_coeff", default=0.0, type=float)
parser.add_argument("--sup_loss_choice", default='multi', choices=["multi", "single"], type=str)
parser.add_argument("--dropout", default=0.3, type=float)
parser.add_argument("--word_dropout", default=0.1, type=float)
parser.add_argument("--l2_reg", default=0, type=float)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--lr_reduction", default=4., type=float)
parser.add_argument("--wait_epochs", default=1, type=float)
parser.add_argument("--save_all", default=True, type=bool)

flags = parser.parse_args()

# Manual Settings, Deactivate before pushing
if False:
    flags.sup_loss_choice = 'single'
    flags.batch_size = 128
    flags.grad_accu = 1
    flags.max_len = 17
    # flags.sup_coeff = 1.0
    flags.dec_sup_coeff = 1.0
    # flags.test_name = "nliLM/SNLIRegular_beta0.4.4"
    flags.test_name = "nliLM/sup_test"
    flags.data = "sup_nli"
    flags.n_latents = [4]
    flags.graph = "IndepInfer"
    # flags.losses = "LagVAE"
    flags.kl_beta = 0.3

    # flags.z_size = 16
    # flags.encoder_h = 256
    # flags.decoder_h = 256


# torch.autograd.set_detect_anomaly(True)
GRAPH = {"Vanilla": get_vanilla_graph,
         "VanillaTr": get_vanilla_Transformer_graph,
         "Discrete": get_discrete_auto_regressive_graph,
         "IndepInfer": get_structured_auto_regressive_indep_graph,
         "Normal": get_structured_auto_regressive_graph,
         "NormalConGen": get_structured_auto_regressive_graphConGen,
         "Normal2": get_structured_auto_regressive_graph2,
         "NormalLSTM": get_lstm_graph,
         "NormalSimplePrior": get_structured_auto_regressive_simple_prior}[flags.graph]
if flags.graph == "NormalLSTM":
    flags.encoder_h = int(flags.encoder_h/k*klstm)
if flags.graph in ("Vanilla", "VanillaTr"):
    flags.n_latents = [flags.z_size]
if flags.losses == "LagVAE":
    flags.anneal_kl0 = 0
    flags.anneal_kl1 = 0
Data = {"nli": NLIGenData2, "ontonotes": OntoGenData, "yelp": HuggingYelp2, "yelp_reg": HuggingYelpReg,
        "de_nli": GermanNLIGenData2, "sup_yelp": SupYelpData, "sup_nli": SupNLIData}[flags.data]
MAX_LEN = flags.max_len
BATCH_SIZE = flags.batch_size
GRAD_ACCU = flags.grad_accu
N_EPOCHS = flags.n_epochs
TEST_FREQ = flags.test_freq
COMPLETE_TEST_FREQ = flags.complete_test_freq
DEVICE = device(flags.device)
SUPERVISED = flags.data.startswith("sup_")
print('This run is {}supervised'.format("" if SUPERVISED else 'not '))
# This prevents illegal memory access on multigpu machines (unresolved issue on torch's github)
if flags.device.startswith('cuda'):
    torch.cuda.set_device(int(flags.device[-1]))
LOSSES = {'IWAE': [IWLBo],
          'VAE': [ELBo],
          'LagVAE': [ELBo]}[flags.losses]

ANNEAL_KL = [flags.anneal_kl0*flags.grad_accu, flags.anneal_kl1*flags.grad_accu]
LOSS_PARAMS = [1]
if flags.grad_accu > 1:
    LOSS_PARAMS = [w/flags.grad_accu for w in LOSS_PARAMS]


def main():
    data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE, pretrained=flags.pretrained_embeddings)
    h_params = HParams(len(data.vocab.itos), len(data.tags.itos) if flags.data == 'yelp' else None, MAX_LEN, BATCH_SIZE, N_EPOCHS,
                       device=DEVICE, vocab_ignore_index=data.vocab.stoi['<pad>'], decoder_h=flags.decoder_h,
                       decoder_l=flags.decoder_l, encoder_h=flags.encoder_h, encoder_l=flags.encoder_l,
                       text_rep_h=flags.text_rep_h, text_rep_l=flags.text_rep_l, n_heads=flags.n_heads,
                       test_name=flags.test_name, grad_accumulation_steps=GRAD_ACCU,
                       optimizer_kwargs={'lr': flags.lr, #'weight_decay': flags.l2_reg, 't0':100, 'lambd':0.},
                                         'weight_decay': flags.l2_reg, 'betas': (0.9, 0.99)},
                       is_weighted=[], graph_generator=GRAPH, no_sa=flags.no_sa, tr_enc_in_dec=flags.tr_enc_in_dec,
                       z_size=flags.z_size, embedding_dim=flags.embedding_dim, anneal_kl=ANNEAL_KL,
                       grad_clip=flags.grad_clip*flags.grad_accu, kl_th=flags.kl_th, highway=flags.highway,
                       losses=LOSSES, sup_loss_choice=flags.sup_loss_choice, dropout=flags.dropout, training_iw_samples=flags.training_iw_samples,
                       testing_iw_samples=flags.testing_iw_samples, loss_params=LOSS_PARAMS, optimizer=optim.AdamW,
                       markovian=flags.markovian, word_dropout=flags.word_dropout, contiguous_lm=False,
                       test_prior_samples=flags.test_prior_samples, n_latents=flags.n_latents,
                       max_elbo=[flags.max_elbo_choice, flags.max_elbo1],  lv_kl_coeff=flags.lv_kl_coeff, sup_coeff=flags.sup_coeff,
                       dec_sup_coeff = flags.dec_sup_coeff,
                       z_emb_dim=flags.z_emb_dim, minimal_enc=flags.minimal_enc, kl_beta=flags.kl_beta)
    val_iterator = iter(data.val_iter)
    print("Words: ", len(data.vocab.itos), ", On device: ", DEVICE.type)
    print("Loss Type: ", flags.losses)
    if flags.losses == 'LagVAE':
        model = LaggingDisentanglementTransformerVAE(data.vocab, data.tags, h_params, wvs=data.wvs, dataset=flags.data,
                                                     enc_iter=data.enc_train_iter)
    else:
        model = DisentanglementTransformerVAE(data.vocab, data.tags, h_params, wvs=data.wvs, dataset=flags.data)
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
    for k, appr in model.infer_bn.approximator.items():
        number_parameters = sum(p.numel() for p in appr.parameters() if p.requires_grad)
        print("-->Inference approximator for {} has {} M params".format(k.name, number_parameters/1e6))
    for k, appr in model.gen_bn.approximator.items():
        number_parameters = sum(p.numel() for p in appr.parameters() if p.requires_grad)
        print("-->Generation approximator for {} has {} M params".format(k.name, number_parameters/1e6))
    print("Generation parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    number_parameters = sum(p.numel() for p in model.word_embeddings.parameters() if p.requires_grad)
    print("Embedding parameters: ", "{0:05.2f} M".format(number_parameters/1e6))
    min_perp = 1e20
    wait_count = 0
    loss = torch.tensor(1e20)
    mean_loss = 0
    stabilize_epochs = 0
    prev_mi = 0
    # model.eval()
    # model.step = 8000
    # print(model.get_disentanglement_summaries2(data.test_iter, 200))
    # print(model.get_perplexity(data.val_iter))
    # print(model.collect_stats(data.val_iter))
    while data.train_iter is not None:  # Add False for eval mode
        for i, training_batch in enumerate(data.train_iter):
            if training_batch.text.shape[1] < 2: continue

            if model.step == h_params.anneal_kl[0]:
                model.optimizer = h_params.optimizer(model.parameters(), **h_params.optimizer_kwargs)
                print('Refreshed optimizer !')
                if model.step != 0 and not torch.isnan(loss):
                    model.save()
                    print('Saved model after it\'s pure reconstruction phase')

            # print([' '.join([data.vocab.itos[t] for t in text_i]) for text_i in training_batch.text[:2]])
            inp = {'x': training_batch.text[..., 1:], 'x_prev': training_batch.text[..., :-1]}
            if SUPERVISED:
                inp['sup'] = training_batch.label
            loss = model.opt_step(inp)

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
                    inp = {'x': test_batch.text[..., 1:], 'x_prev': test_batch.text[..., :-1]}
                    if SUPERVISED:
                        inp['sup'] = test_batch.label
                    model(inp)
                model.dump_test_viz(complete=int(model.step / (len(LOSSES))) %
                                    COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)
                model.train()
            if model.step >= 7000:
                h_params.max_elbo = [flags.max_elbo_choice, flags.max_elbo2]
            current_time = time()
        data.reinit_iterator('valid')
        if model.step >= h_params.anneal_kl[0]:  # and ((data.n_epochs % 3) == 0):
            model.eval()
            pp_ub = 0.0  # model.get_perplexity(data.val_iter)
            print("perplexity is {} ".format(pp_ub))
            if flags.data == "yelp":
                max_auc, auc_margin, max_auc_index  = model.get_sentiment_summaries(data.val_iter)
                print("max_auc: {}, auc_margin: {}, max_auc_index: {} ".format(max_auc, auc_margin, max_auc_index))
            # else:
            # dis_diffs1, dis_diffs2, _, _ = model.get_disentanglement_summaries()
            # print("disentanglement scores : {} and {}".format(dis_diffs1, dis_diffs2))
            val_dec_lab_wise_disent, val_enc_lab_wise_disent, \
            val_decoder_Ndisent_vars, val_encoder_Ndisent_vars\
                = model.get_disentanglement_summaries2(data.val_iter, 200)
            print("Encoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_enc_lab_wise_disent,
                                                                           sum(val_enc_lab_wise_disent.values()),
                                                                                      val_encoder_Ndisent_vars))
            print("Decoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_dec_lab_wise_disent,
                                                                           sum(val_dec_lab_wise_disent.values()),
                                                                                      val_decoder_Ndisent_vars))

            # print("Perplexity Upper Bound is {} at step {}".format(pp_ub, model.step))
            data.reinit_iterator('valid')

            dev_kl, dev_kl_std, dev_rec, val_mi = model.collect_stats(data.val_iter)
            data.reinit_iterator('valid')
            if val_mi < prev_mi and flags.losses == "LagVAE":
                print("Stopped aggressive training phase")
                model.aggressive = False
            prev_mi = val_mi

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
    print("================= Finished training : Getting Scores on test set ============")
    model.eval()
    #
    # val_dec_lab_wise_disent, val_dec_lab_wise_disent_recon, val_enc_lab_wise_disent, \
    # val_decoder_Ndisent_vars, val_decoder_Ndisent_vars_recon, val_encoder_Ndisent_vars \
    #     = model.get_disentanglement_summaries3(data.val_iter, n_samples=2000)
    val_dec_lab_wise_disent, val_enc_lab_wise_disent, val_decoder_Ndisent_vars, val_encoder_Ndisent_vars \
        = model.get_disentanglement_summaries2(data.val_iter, n_samples=2000)
    print("Encoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_enc_lab_wise_disent,
                                                                   sum(val_enc_lab_wise_disent.values()),
                                                                              val_encoder_Ndisent_vars))
    print("Decoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_dec_lab_wise_disent,
                                                                   sum(val_dec_lab_wise_disent.values()),
                                                                              val_decoder_Ndisent_vars))
    # print("Decoder Reconstruction Disentanglement Scores : {}, Total : {}, Nvars: {}".format(val_dec_lab_wise_disent_recon,
    #                                                                sum(val_dec_lab_wise_disent_recon.values()),
    #                                                                           val_decoder_Ndisent_vars_recon))
    # test_dec_lab_wise_disent, test_dec_lab_wise_disent_recon, test_enc_lab_wise_disent, \
    # test_decoder_Ndisent_vars, test_decoder_Ndisent_vars_recon, test_encoder_Ndisent_vars\
    #     = model.get_disentanglement_summaries3(data.test_iter, n_samples=2000)
    test_dec_lab_wise_disent, test_enc_lab_wise_disent, test_decoder_Ndisent_vars, test_encoder_Ndisent_vars\
        = model.get_disentanglement_summaries2(data.test_iter, n_samples=2000)
    data.reinit_iterator('test')
    print("Encoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(test_enc_lab_wise_disent,
                                                                   sum(test_enc_lab_wise_disent.values()),
                                                                              test_encoder_Ndisent_vars))
    print("Decoder Disentanglement Scores : {}, Total : {}, Nvars: {}".format(test_dec_lab_wise_disent,
                                                                   sum(test_dec_lab_wise_disent.values()),
                                                                              test_decoder_Ndisent_vars))
    # print("Decoder Reconstruction Disentanglement Scores : {}, Total : {}, Nvars: {}".format(test_dec_lab_wise_disent_recon,
    #                                                                sum(test_dec_lab_wise_disent_recon.values()),
    #                                                                           test_decoder_Ndisent_vars_recon))
    pp_ub = model.get_perplexity(data.val_iter)
    test_pp_ub = model.get_perplexity(data.test_iter)
    print("Perplexity: {}".format(test_pp_ub))
    dev_kl, dev_kl_std, dev_rec, val_mi = model.collect_stats(data.val_iter)
    test_kl, test_kl_std, test_rec, test_mi = model.collect_stats(data.test_iter)
    # relations = ["nsubj", "verb", "obj", "iobj"]
    relations = ['nsubj', 'verb', 'dobj', 'pobj'] if flags.data != "de_nli" else ['sb', 'verb', 'oa', 'da', 'op', 'oc']
    temps = ['syntemp', 'lextemp']
    if not os.path.exists(flags.csv_out):
        with open(flags.csv_out, 'w') as f:
            f.write('\t'.join(['name', 'net_size', 'z_size', 'graph', 'data', 'kl_beta', 'n_latents',
                               'dev_kl', 'dev_kl_std', 'dev_ppl', 'dev_tot_dec_disent', 'dev_tot_dec_disent_recon',
                              'dev_tot_en_disent', *['dev_dec_disent_'+r for r in relations],
                               *['dev_dec_disent_recon_'+r for r in relations],
                               'dev_dec_disent_syntemp', 'dev_dec_disent_lextemp',
                               *['dev_enc_disent_' + r for r in relations],
                                'dev_rec_error', 'dev_decoder_Ndisent_vars', 'dev_encoder_Ndisent_vars',
                              'test_kl', 'test_kl_std', 'test_ppl', 'test_tot_dec_disent', 'test_tot_dec_disent_recon',
                              'test_tot_en_disent', 'test_dec_disent_subj', *['test_dec_disent_'+r for r in relations],
                               *['test_dec_disent_recon_' + r for r in relations],
                              'test_dec_disent_syntemp', 'test_dec_disent_lextemp', *['test_enc_disent_'+r for r in relations],
                                'test_rec_error', 'test_decoder_Ndisent_vars',
                               'test_encoder_Ndisent_vars',
                              'dev_mi', 'test_mi'])+'\n')
    with open(flags.csv_out, 'a') as f:
        f.write('\t'.join([flags.test_name, str(flags.encoder_h), str(flags.z_size), str(flags.graph), str(flags.data),
                           str(flags.kl_beta), str(flags.n_latents),
                           str(dev_kl), str(dev_kl_std), str(pp_ub), str(sum(val_dec_lab_wise_disent.values())),
                           # str(sum(val_dec_lab_wise_disent_recon.values())),
                           str(sum(val_enc_lab_wise_disent.values())),
                           *[str(val_dec_lab_wise_disent[k]) for k in relations+temps],
                           # *[str(val_dec_lab_wise_disent_recon[k]) for k in relations+temps],
                           *[str(val_enc_lab_wise_disent[k]) for k in relations], str(dev_rec),
                           str(val_decoder_Ndisent_vars), str(val_encoder_Ndisent_vars),
                           str(test_kl), str(test_kl_std), str(test_pp_ub), str(sum(test_dec_lab_wise_disent.values())),
                           # str(sum(test_dec_lab_wise_disent_recon.values())),
                           str(sum(test_enc_lab_wise_disent.values())),
                           *[str(test_dec_lab_wise_disent[k]) for k in relations+temps],
                           # *[str(test_dec_lab_wise_disent_recon[k]) for k in relations+temps],
                           *[str(test_enc_lab_wise_disent[k]) for k in relations], str(test_rec),
                           str(test_decoder_Ndisent_vars), str(test_encoder_Ndisent_vars), str(val_mi), str(test_mi)
                         ])+'\n')


    print("Finished training !")


def limited_next(iterator):
    batch = next(iterator)
    if len(batch.text[0]) > MAX_LEN:
        batch.text = batch.text[:, :MAX_LEN]
        batch.label = batch.label[:, :MAX_LEN-2]
    return batch


if __name__ == '__main__':
    main()


