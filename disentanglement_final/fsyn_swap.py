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
    BARTParaNMT, BARTNLI, BARTNewsCategory, BARTFrSbt, BARTWiki, BARTBookCorpus, BARTOpenWT
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
                                                      'bc', 'owt'], type=str)
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
parser.add_argument("--bart_l", default=None, type=int or None)#################"
parser.add_argument("--aux_l", default=2, type=int or None)#################"
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

if True:
    # flags.optimizer="sgd"
    flags.use_bart = True
    flags.layer_wise_qkv = True
    flags.batch_size = 10
    flags.max_len = 20
    #flags.test_name = "Disentanglement2/ParaQKVBARTminiLag_beta0.4.0.4.1.8"
    #flags.test_name = "Disentanglement2/ParaQKVBARTminiThr_beta0.2.0.2.1.8"
    #flags.test_name = "Disentanglement2/BCQBmTX_beta0.6.0.6.1.4"
    flags.test_name = "Disentanglement2/PQBmTXLmini64_beta0.6.0.6.1.4"
    #flags.test_name = "Disentanglement2/WikiXNoDrop_beta0.3.0.3.1.4"
    flags.data = "owt"
    #flags.data = "bc"
    flags.bart_l = 3
    flags.n_latents = [4]
    flags.graph = "NQKV"
    flags.kl_beta = 0.3
    flags.max_elbo_choice = 6
    flags.kl_beta_zs = 0.3
    flags.z_size = 768#192
    ML, BS, NS, NAS = 20, 20, 5, 3 # GenLen, Beam Size, Gen NSamples, NAlternative samples
if flags.use_bart:
    flags.decoder_h = 768
    flags.encoder_h = 768
    flags.embedding_dim = 768

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
        "news": BARTNewsCategory, "wiki": BARTWiki, "bc": BARTBookCorpus, "owt": BARTOpenWT,
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
                   fr=flags.data == 'fr_sbt', bart_l=flags.bart_l, aux_l=flags.aux_l)
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

model.eval()
model.gen_bn.clear_values(), model.infer_bn.clear_values(), torch.cuda.empty_cache()
with torch.no_grad():
    model.beam_size = BS
    #decoder_syn_disent_scores = model.get_swap_tma(n_samples=240, batch_size=12, beam_size=BS)
    #print(decoder_syn_disent_scores)
    text, samples, params = model.get_sentences(NS, gen_len=ML, sample_w=False, vary_z=True, complete=None,
                                                contains=None, max_tries=100)
    alt_text, alt_samples = model._get_alternative_sentences(samples, params, [sum(h_params.n_latents)], NAS, ML, complete=None)
    print("====== Changing Structure=======", flush=True)
    for i in range(len(text)):
        print("-->", text[i], '|<', '><'.join(alt_text[i::len(text)]), '>', flush=True)
    alt_text, alt_samples = model._get_alternative_sentences(samples, params, list(range(sum(h_params.n_latents))), NAS, ML, complete=None)
    print("====== Changing Content=======")
    for i in range(len(text)):
        print("-->", text[i], '|<', '><'.join(alt_text[i::len(text)]), '>', flush=True)


    def swap_syntax(mdl, prev_latent_vals, syn_src_lvs):
        assert 'zs' in mdl.gen_bn.name_to_v
        has_zg = 'zg' in mdl.gen_bn.name_to_v

        n_orig_sentences = prev_latent_vals['z1'].shape[0]
        n_samples = 1
        go_symbol = torch.ones([n_samples * n_orig_sentences]).long() * \
                    mdl.index[mdl.generated_v].stoi[mdl.go_symbol]
        go_symbol = go_symbol.to(mdl.h_params.device).unsqueeze(-1)
        x_prev = go_symbol
        orig_z = prev_latent_vals['z1']
        orig_zst = syn_src_lvs['zs']
        if has_zg:
            orig_zg = prev_latent_vals['zg'].unsqueeze(1).repeat(1, n_samples, 1)
            orig_zg = orig_zg.transpose(0, 1).reshape(n_samples * n_orig_sentences, -1)

        z_input = {'z1': orig_z.unsqueeze(1), **({'zs': orig_zst.unsqueeze(1)}),
                   **({'zg': orig_zg.unsqueeze(1)} if has_zg else {})}

        x_prev, seq_scores = mdl.generate_from_z2(z_input, x_prev, beam_size=mdl.beam_size, return_seq_scores=True)
        text = mdl.decode_to_text2(x_prev, mdl.h_params.vocab_size, mdl.index[mdl.generated_v])
        return text, {'z1': orig_z}, seq_scores


    sw_zs = [sum(h_params.n_latents)]
    model.infer_bn.clear_values(), model.gen_bn.clear_values()
    torch.cuda.empty_cache()
    model.beam_size = BS
    print("========== BEAM SIZE: {} ==================".format(model.beam_size), flush=True)
    semp
    sw_text, sw_samples, _ = swap_syntax(model, samples)
    print(text, flush=True)
    for i in range(len(text)):
        for j in range(len(text)):
            if i != j:
                print("z_from: ", text[i], "|z_to: ", text[j], "|result: ", sw_text[len(text) * i + j], flush=True)
    model.beam_size = 1
    print("========== BEAM SIZE: {} ==================".format(model.beam_size), flush=True)
    sw_text, sw_samples, _ = swap_syntax(model, samples)
    print(text, flush=True)
    for i in range(len(text)):
        for j in range(len(text)):
            if i != j:
                print("z_from: ", text[i], "|z_to: ", text[j], "|result: ", sw_text[len(text) * i + j], flush=True)
    model.beam_size = BS

    print("========== Enc/Dec Part ==================", flush=True)
    sents = [
    "the eyes of his teammates had turned ugly and hostile .",
    "you wan na grab some dinner ?",
    "imagine how delighted i must have been , and how surprised .",
    "there 'll be audience having laughs and entertainment .",
    "all things have their destiny .",
    # "we can schedule you for that in four to six weeks .",
    # "this is who you were supposed to be .",
    # "moore starting to move on after durellea in round four .",
    # "the blunderbuss blast of the russian hand-cannon had lacerated the men hideously .",
    # "two veruca shows in two nights ?",
    # "you guys met linda bloom at the barbecue ?",
    # "we 're not inviting d'hoffyna .",
    # "we 've altered course .",
    # "let 's sleep .",
    ]

    ezs, ezc = model.embed_sents(sents)
    enc_samples = {"z1":ezc, "zs":ezs, "zg":torch.zeros_like(ezs)}
    print("========== Reconstructions==================", flush=True)
    sw_zs = [sum(h_params.n_latents)]
    sw_text, sw_samples, = swap_syntax(model, enc_samples)
    for i in range(len(sents)):
        print("z_from: ", sents[i], "|z_to: ", sents[i], "|result: ", sw_text[len(sents) * i + i], flush=True)
    print("========== Transfer==================", flush=True)
    print(sw_text, flush=True)
    for i in range(len(sents)):
        for j in range(len(sents)):
            if i != j:
                print("z_from: ", sents[i], "|z_to: ", sents[j], "|result: ", sw_text[len(sents) * i + j], flush=True)
    print("========== Syntax Swap==================", flush=True)

    sents1 = ["his teammates ' eyes got an ugly , hostile expression .",
              "do you want to go to dinner ?",
              "just judge how happy i was to be surprised !",
              "the audience will laugh and have fun .",
              "there is a destiny for everything .",
              "you can be scheduled for that in four to six weeks .",
              "you should be like this man .",
              "in the fourth round , moore begins attacking durellea  .",
              "the man was lacerated hideously by the blunderbuss blast of the russian hand-cannon .",
              "there are two veruca in two nights ?",
              "did you meet linda bloom on a barbecue ?"]

    sents2 = ["the smell of flowers was thick and sweet .",
              "you do n't like playing the martyr ?",
              "describe how exciting it could be , and how heartbreaking .",
              "there 'll be bandits waiting and robbing banks .",
              "every family wants a baby .",
              "i 'll meet you out there in 15 minutes .",
              "that 's what we 're supposed to find out .",
              "y'all gon na get on out of here by tomorrow .",
              "the homicide rate in this country had reached epidemic proportions .",
              "air support arrives in 2 minutes ?",
              "you lost your family in a fire ?"]

    ezs, ezc = model.embed_sents(sents1)
    enc_samples1 = {"z1": ezc, "zs": ezs, "zg":torch.zeros_like(ezs)}
    ezs, ezc = model.embed_sents(sents2)
    enc_samples2 = {"z1": ezc, "zs": ezs, "zg":torch.zeros_like(ezs)}

    rec_text1, _, _ = swap_syntax(model, enc_samples1, enc_samples1)
    rec_text2, _, _ = swap_syntax(model, enc_samples2, enc_samples2)
    sw_text, _, seqsco = swap_syntax(model, enc_samples1, enc_samples2)
    for i in range(len(sw_text)):
        print("=========== example {} ==========".format(i))
        print("cont_src :[", sents1[i], "], syn_src: [", sents2[i])
        print("rec_cont_src :[", rec_text1[i], "], rec_syn_src: [", rec_text2[i], "]")
        print(" result rec (sc={}):> ".format(seqsco[i]), sw_text[i], flush=True)

    eval_file = os.path.join('.data', 'paranmt2', 'test_input.txt')
    sents1, sents2 = [], []
    res_sens = []
    with open(eval_file, 'r') as f:
        for line in f:
            l_sens = line.split('\t')
            sents1.append(l_sens[0]), sents2.append(l_sens[1][:-1])
            if len(sents1) == 10:
                ezs, ezc = model.embed_sents(sents1)
                enc_samples1 = {"z1": ezc, "zs": ezs, "zg":torch.zeros_like(ezs)}
                ezs, ezc = model.embed_sents(sents2)
                enc_samples2 = {"z1": ezc, "zs": ezs, "zg":torch.zeros_like(ezs)}
                sw_text, _, seqsco = swap_syntax(model, enc_samples1, enc_samples2)
                res_sens.extend(sw_text)
                sents1, sents2 = [], []
                print("swapped {} senteces".format(len(res_sens)))

        if len(sents1) > 0:
            ezs, ezc = model.embed_sents(sents1)
            enc_samples1 = {"z1": ezc, "zs": ezs, "zg": torch.zeros_like(ezs)}
            ezs, ezc = model.embed_sents(sents2)
            enc_samples2 = {"z1": ezc, "zs": ezs, "zg": torch.zeros_like(ezs)}
            sw_text, _, seqsco = swap_syntax(model, enc_samples1, enc_samples2)
            res_sens.extend(sw_text)
    res_file = os.path.join(os.path.split(h_params.test_name)[-1]+"_synswap_testres.txt")
    with open(res_file, 'w') as f:
        for sen in res_sens:
            f.write(sen+'\n')
