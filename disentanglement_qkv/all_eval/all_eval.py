import os
from time import time
import numpy as np
import torch

#from datasets import load_metric
#meteor = load_metric("meteor")

from nltk.translate import meteor_score

from transformers import BartTokenizer, BartConfig
from tqdm import tqdm
from supar import Parser
from scipy.stats import ttest_ind

from parabart import ParaBart

from eval_utils import Meteor, stanford_parsetree_extractor, \
    compute_tree_edit_distance
meteor = Meteor()
# Must have in the same folder the "evaluation" folder of VGAVAE, the parabart.py script and its model folder, and the transformers folder
const_parser = Parser.load('crf-con-en')

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--input_file', '-i', type=str)
# parser.add_argument('--ref_file', '-r', type=str)
# args = parser.parse_args()

TEST = True

if TEST:
    root = os.path.join('..', 'trans_out', 'test_files')
    inp_f = os.path.join(root, 'test_input.txt')
    sem_f = os.path.join(root, 'sem_ref.txt')
    syn_f = os.path.join(root, 'syn_ref.txt')
    para_f = os.path.join(root, 'test_ref.txt')
else:
    root = os.path.join('..', 'trans_out', 'dev_files')
    inp_f = os.path.join(root, 'dev_input.txt')
    sem_f = os.path.join(root, 'sem_ref.txt')
    syn_f = os.path.join(root, 'syn_ref.txt')
    para_f = os.path.join(root, 'dev_ref.txt')

arg_f = "eval_args.txt"
global res_f
syn_parses = None
sem_parses = None
para_parses = None
res_fs = [f.strip() for f in open(arg_f).readlines()]
# res_f = os.path.join(root, 'PQBmTXLmini64_2_beta0.6.0.6.1.4_synswap_res.txt')


def get_sents_from_file(path, codec=False):
    if path is None: return None
    if codec: op = lambda x: open(x, encoding="UTF-8")
    else: op=open
    with op(path) as f:
        sens = []
        for line in f:
            sens.append(line[:-1])
    return sens


def truncate_tree(tree, lv):
    tok_i = 0
    curr_lv = 0
    tree_toks = tree.split()
    while tok_i != len(tree_toks):
        if tree_toks[tok_i].startswith('('):
            curr_lv += 1
        else:
            closed_lvs = int(tree_toks[tok_i].count(')'))
            if curr_lv - closed_lvs <= lv:
                tree_toks[tok_i] = ')'*(closed_lvs - (curr_lv-lv))
            curr_lv -= closed_lvs
        if lv >= curr_lv and tree_toks[tok_i]!='':
            tok_i += 1
        else:
            tree_toks.pop(tok_i)
    return ' '.join(tree_toks)


def get_lin_parse_tree(sens):
    tree_parses = const_parser.predict(sens, lang='en', verbose=False)
    lin_parses = []
    for p in tree_parses:
        lin_p = repr(p)
        if lin_p.startswith("(TOP"):
            lin_p = lin_p[5:-1]
        lin_parses.append(lin_p)
    return lin_parses


def template_match(l1, l2, lv, verbose=0, filter_empty=True):
    if filter_empty:
        not_empty1 = [any([c != " " for c in li1]) for li1 in l1]
        not_empty2 = [any([c != " " for c in li2]) for li2 in l2]
        l1 = [li1 for li1, ne1, ne2 in zip(l1, not_empty1, not_empty2) if ne1 and ne2]
        l2 = [li2 for li2, ne1, ne2 in zip(l2, not_empty1, not_empty2) if ne1 and ne2]
    docs1, docs2 = get_lin_parse_tree(l1), get_lin_parse_tree(l2)
    temps1 = [truncate_tree(doc, lv) for doc in docs1]
    temps2 = [truncate_tree(doc, lv) for doc in docs2]
    if verbose:
        for l, t in zip(l1+l2, temps1+temps2):
            print(l, "-->", t)
        print("+++++++++++++++++++++++++")
    return [int(t1 == t2) for t1, t2 in zip(temps1, temps2)]


def build_embeddings(model, tokenizer, sents, name):
    model.eval()
    embeddings = torch.ones((len(sents), model.config.d_model))
    with torch.no_grad():
        for i, sent in tqdm(enumerate(sents), desc="Getting {} Embeddings".format(name)):
            sent_inputs = tokenizer(sent, return_tensors="pt")
            sent_token_ids = sent_inputs['input_ids']
            sent_embed = model.encoder.embed(sent_token_ids.cuda())
            embeddings[i] = sent_embed.detach().cpu().clone()
    return embeddings


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def get_parabart_eval():
    print("==== loading ParaBart model ====")
    config = BartConfig.from_pretrained('facebook/bart-base', cache_dir='../para-data/bart-base')
    model = ParaBart(config)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir='../para-data/bart-base')
    model.load_state_dict(torch.load("./model/model.pt", map_location='cpu'))
    # model = model.to('cpu')
    model = model.cuda()
    # path_result = os.path.join('..', '..', '..', 'TSE', 'VGVAE', 'test_files', 'advaelvres.txt')
    syn_source_sents, sem_source_sents, ref_sents, result_sents = [], [], [], []
    with open(inp_f, "r", encoding="UTF-8") as f:
        for l in f:
            spl = l[:-1].split('\t')
            sem_source_sents.append(spl[0])
            syn_source_sents.append(spl[1])
    with open(res_f, "r", encoding="UTF-8"
              ) as f:
        for l in f:
            spl = l[:-1].split('\t')
            result_sents.append(spl[0])
    with open(para_f, "r", encoding="UTF-8") as f:
        for l in f:
            spl = l[:-1].split('\t')
            ref_sents.append(spl[0])
    sem_source_embs = build_embeddings(model, tokenizer, sem_source_sents, "sem_source")
    syn_source_embs = build_embeddings(model, tokenizer, syn_source_sents, "syn_source")
    ref_embs = build_embeddings(model, tokenizer, ref_sents, "ref")
    result_embs = build_embeddings(model, tokenizer, result_sents, "result")
    print("Calculating similarities")
    Syn2sem_sims = np.array([cosine(s, t) for s, t in zip(sem_source_embs, syn_source_embs)])
    Syn2ref_sims = np.array([cosine(s, t) for s, t in zip(syn_source_embs, ref_embs)])
    Sem2ref_sims = np.array([cosine(s, t) for s, t in zip(sem_source_embs, ref_embs)])
    sem_res_sims = np.array([cosine(s, t) for s, t in zip(sem_source_embs, result_embs)])
    syn_res_sims = np.array([cosine(s, t) for s, t in zip(syn_source_embs, result_embs)])
    para_sims = np.array([cosine(s, t) for s, t in zip(result_embs, ref_embs)])

    print("=========== Similarities ===================")
    print("Syn2sem similarity: ", np.mean(Syn2sem_sims), np.std(Syn2sem_sims))
    print("Syn2ref similarity: ", np.mean(Syn2ref_sims), np.std(Syn2ref_sims))
    print("Sem2ref similarity: ", np.mean(Sem2ref_sims), np.std(Sem2ref_sims))
    print("Syntax transfer semantic similarity: ", np.mean(syn_res_sims), np.std(syn_res_sims))
    print("Content transfer semantic similarity: ", np.mean(sem_res_sims), np.std(sem_res_sims))
    print("Content transfer semantic similarity with paraphrase: ", np.mean(para_sims), np.std(para_sims))
    print("Similarity accuracy: ", np.mean(sem_res_sims > syn_res_sims))
    print("Parabart Similarity accuracy: ", np.mean(Sem2ref_sims > Syn2ref_sims))
    PB_pval = ttest_ind(syn_res_sims, sem_res_sims)[1]
    return np.mean(syn_res_sims)*100, np.mean(sem_res_sims)*100, np.mean(para_sims)*100, PB_pval


def get_tma():
    print("Getting sentences")
    syn_src_sens, sem_src_sens, para_sens, res_sens = get_sents_from_file(syn_f), get_sents_from_file(sem_f),\
                                                      get_sents_from_file(para_f), get_sents_from_file(res_f, codec=True)

    syn2res_tma2, syn2res_tma3 = template_match(syn_src_sens, res_sens, 2), \
                                   template_match(syn_src_sens, res_sens, 3)
    sem2res_tma2, sem2res_tma3 = template_match(sem_src_sens, res_sens, 2), \
                                   template_match(sem_src_sens, res_sens, 3)
    para2res_tma2, para2res_tma3 = template_match(para_sens, res_sens, 2), \
                                   template_match(para_sens, res_sens, 3)
    tma2_pval, tma3_pval = ttest_ind(syn2res_tma2, sem2res_tma2)[1], ttest_ind(syn2res_tma3, sem2res_tma3)[1]
    syn2res_tma2 = np.mean(syn2res_tma2) * 100
    syn2res_tma3 = np.mean(syn2res_tma3) * 100
    sem2res_tma2 = np.mean(sem2res_tma2) * 100
    sem2res_tma3 = np.mean(sem2res_tma3) * 100
    para2res_tma2 = np.mean(para2res_tma2) * 100
    para2res_tma3 = np.mean(para2res_tma3) * 100
    print("syn_tma2:{}, syn_tma3:{}".format(syn2res_tma2, syn2res_tma3))
    print("sem_tma2:{}, sem_tma3:{}".format(sem2res_tma2, sem2res_tma3))
    print("para_tma2:{}, para_tma3:{}".format(para2res_tma2, para2res_tma3))
    return syn2res_tma2, syn2res_tma3, sem2res_tma2, sem2res_tma3, para2res_tma2, para2res_tma3, tma2_pval, tma3_pval


def get_STD(ref_parses, input_parses):
    assert len(input_parses) == len(ref_parses)
    all_ted = []
    pbar = tqdm(zip(input_parses, ref_parses))

    for input_parse, ref_parse in pbar:
        ted = compute_tree_edit_distance(input_parse, ref_parse)
        all_ted.append(ted)
        pbar.set_description("syntax-TED: {:.3f}".format(sum(all_ted) / len(all_ted)))

    print("syntax-TED: {:.3f}".format(sum(all_ted) / len(all_ted)))
    return all_ted


def get_all_M_STD():

    # Getting Meteor
    # meteor.add_batch(predictions=open(syn_f).readlines(), references=open(res_f).readlines())
    # syn_M = meteor.compute()['meteor']*100
    # meteor.add_batch(predictions=open(sem_f).readlines(), references=open(res_f).readlines())
    # sem_M = meteor.compute()['meteor']*100
    # meteor.add_batch(predictions=open(para_f).readlines(), references=open(res_f).readlines())
    # para_M = meteor.compute()['meteor']*100
    syn_M = [meteor_score.single_meteor_score(ref, pred) for pred, ref in zip(open(res_f).readlines(), open(syn_f).readlines())]
    sem_M = [meteor_score.single_meteor_score(ref, pred) for pred, ref in zip(open(res_f).readlines(), open(sem_f).readlines())]
    para_M = [meteor_score.single_meteor_score(ref, pred) for pred, ref in zip(open(res_f).readlines(), open(para_f).readlines())]
    # for pred, ref in zip(open(res_f).readlines(), open(syn_f).readlines()):
    #     print(pred, ref)
    #     meteor._score(pred.strip(), [ref.strip()])
    # syn_M = [meteor._score(pred.strip(), [ref.strip()]) for pred, ref in zip(open(res_f).readlines(), open(syn_f).readlines())]
    # sem_M = [meteor._score(pred.strip(), [ref.strip()]) for pred, ref in zip(open(res_f).readlines(), open(sem_f).readlines())]
    # para_M = [meteor._score(pred.strip(), [ref.strip()]) for pred, ref in zip(open(res_f).readlines(), open(para_f).readlines())]

    # Getting STD
    spe = stanford_parsetree_extractor(french=False)
    input_parses = spe.run(res_f)
    spe.cleanup()
    global syn_parses, sem_parses, para_parses
    syn_parses = spe.run(syn_f) if syn_parses is None else syn_parses
    spe.cleanup()
    sem_parses = spe.run(sem_f) if sem_parses is None else sem_parses
    spe.cleanup()
    para_parses = spe.run(para_f) if para_parses is None else para_parses
    spe.cleanup()
    syn_STD = get_STD(syn_parses, input_parses)
    sem_STD = get_STD(sem_parses, input_parses)
    para_STD = get_STD(para_parses, input_parses)


    # Getting p_values and averaging both
    STD_pval = ttest_ind(syn_STD, sem_STD)[1]
    M_pval = ttest_ind(syn_M, sem_M)[1]
    syn_M, syn_STD = np.mean(syn_M)*100, np.mean(syn_STD)
    sem_M, sem_STD = np.mean(sem_M)*100, np.mean(sem_STD)
    para_M, para_STD = np.mean(para_M)*100, np.mean(para_STD)
    return syn_M, syn_STD, sem_M, sem_STD, para_M, para_STD, M_pval, STD_pval

ti = time()

numbers = []
pvals = []
for file in res_fs:
    print("="*139)
    print(" Processing File {} ".format(file), flush=True)
    print("="*139)
    res_f = file
    t0 = time()
    print("############### Getting Parabart Evaluation ##############", flush=True)
    syn_PB, sem_PB, para_PB, PB_pval = get_parabart_eval()
    t1 = time()
    print("############### Parabart Evaluation took {} seconds ##############".format(t1-t0), flush=True)

    print("############### Getting TMA Evaluation ##############", flush=True)
    syn_tma2, syn_tma3, sem_tma2, sem_tma3, para_tma2, para_tma3, tma2_pval, tma3_pval = get_tma()
    t2 = time()
    print("############### TMA Evaluation took {} seconds ##############".format(t2-t1), flush=True)

    print("############### Getting Meteor and Syntactic Tree Edit Distance Evaluation ##############", flush=True)
    syn_M, syn_STD, sem_M, sem_STD, para_M, para_STD, M_pval, STD_pval = get_all_M_STD()
    t3 = time()
    print("############### Meteor and Syntactic Tree Edit Distance Evaluation took {} seconds ##############".format(t3-t2), flush=True)
    print("############### The whole Evaluation script for this file took {} seconds ##############".format(t3-t0), flush=True)
    numbers.append([syn_M, syn_PB, syn_STD, syn_tma2, syn_tma3, sem_M, sem_PB, sem_STD, sem_tma2, sem_tma3, para_M,
                    para_PB, para_STD, para_tma2, para_tma3])
    pvals.append([0.0]*5+[M_pval, PB_pval, STD_pval, tma2_pval, tma3_pval]+[0.0]*5)
avg = [sum([numbers[i][j]/len(numbers) for i in range(len(numbers))]) for j in range(len(numbers[0]))]
tf = time()
print("############### The whole Evaluation script for all files took {} seconds ##############".format(tf-ti), flush=True)
print("="*65, " Final Results ", "="*65)
print("{:^5s}|||{:^45s}|||{:^45s}|||{:^45s}".format("File", "syn_src", "sem_src", "para"))
print("{:^5}|||{:^8}|{:^8}||{:^8}|{:^8}|{:^8}|||{:^8}|{:^8}||{:^8}|{:^8}|{:^8}|||{:^8}|"
      "{:^8}||{:^8}|{:^8}|{:^8}".format("",
                                        "M", "PB", "STD", "TMA2", "TMA3",
                                        "M", "PB", "STD", "TMA2", "TMA3",
                                        "M", "PB", "STD", "TMA2", "TMA3"))

for i, (n, pv) in enumerate(zip(numbers, pvals)):
    print("{:^5}|||{:^8.2f}|{:^8.2f}||{:^8.2f}|{:^8.2f}|{:^8.2f}|||{:^8.2f}|{:^8.2f}||{:^8.2f}|{:^8.2f}|{:^8.2f}|||"
          "{:^8.2f}|{:^8.2f}||{:^8.2f}|{:^8.2f}|{:^8.2f}".format(i, *n))
    print("{:^5}|||{:^8.2e}|{:^8.2e}||{:^8.2e}|{:^8.2e}|{:^8.2e}|||{:^8.2e}|{:^8.2e}||{:^8.2e}|{:^8.2e}|{:^8.2e}|||"
          "{:^8.2e}|{:^8.2e}||{:^8.2e}|{:^8.2e}|{:^8.2e}".format('', *pv))
print("-" * 147)
print("{:^5}|||{:^8.2f}|{:^8.2f}||{:^8.2f}|{:^8.2f}|{:^8.2f}|||{:^8.2f}|{:^8.2f}||{:^8.2f}|{:^8.2f}|{:^8.2f}|||"
      "{:^8.2f}|{:^8.2f}||{:^8.2f}|{:^8.2f}|{:^8.2f}".format('avg', *avg))
print("="*147)
print("Files:")
for i, f in enumerate(res_fs):
    print("{}:\t{}".format(i, f))
