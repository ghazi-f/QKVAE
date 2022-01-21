import numpy as np
from supar import Parser
import os
from tqdm import tqdm
FR = False
TEST = True
if FR:
    root = os.path.join('..', '..', '.data', 'fr_sbt')
    sem_f = os.path.join(root, 'sem_ref.txt')
    syn_f = os.path.join(root, 'syn_ref.txt')
    para_f = None
elif TEST:
    root = os.path.join('..', '..', '..', 'TSE', 'VGVAE', 'test_files')
    sem_f = os.path.join(root, 'sem_ref.txt')
    syn_f = os.path.join(root, 'syn_ref.txt')
    para_f = os.path.join(root, 'test_ref.txt')
else:
    root = os.path.join('..', '..', '..', 'TSE', 'VGVAE')
    sem_f = os.path.join(root, 'sem_ref.txt')
    syn_f = os.path.join(root, 'syn_ref.txt')
    para_f = os.path.join(root, 'dev_ref.txt')
# res_f = os.path.join('..', '..', '..', 'TSE', 'VGVAE', 'dev_files', 'advaelvres.txt')
res_f = os.path.join('..', '..', '..', 'TSE', 'VGVAE', 'test_files', 'advaelvres.txt')


def get_sents_from_file(path, codec=False):
    if path is None: return None
    if codec: op = lambda x: open(x, encoding="UTF-8")
    else: op=open
    with op(path) as f:
        sens = []
        for line in f:
            sens.append(line[:-1])
    return sens


print("Getting sentences")
syn_src_sens, sem_src_sens, para_sens, res_sens = get_sents_from_file(syn_f), get_sents_from_file(sem_f),\
                                                  get_sents_from_file(para_f), get_sents_from_file(res_f, codec=True)
if FR:
    const_parser = Parser.load('crf-con-xlmr')
else:
    const_parser = Parser.load('crf-con-en')


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
    if FR:
        batch_size= 20
        docs1, docs2 = [], []
        assert len(l1) % batch_size == 0
        for i in tqdm(range(int(len(l1)/batch_size))):
            docs1.extend(get_lin_parse_tree(l1[i*batch_size:(i+1)*batch_size]))
            docs2.extend(get_lin_parse_tree(l2[i*batch_size:(i+1)*batch_size]))
    else:
        docs1, docs2 = get_lin_parse_tree(l1), get_lin_parse_tree(l2)
    temps1 = [truncate_tree(doc, lv) for doc in docs1]
    temps2 = [truncate_tree(doc, lv) for doc in docs2]
    if verbose:
        for l, t in zip(l1+l2, temps1+temps2):
            print(l, "-->", t)
        print("+++++++++++++++++++++++++")
    return [int(t1 == t2) for t1, t2 in zip(temps1, temps2)]


# print("Calculating various template match metrics")
# syn2sem_tma2, syn2sem_tma3 = np.mean(template_match(syn_src_sens, sem_src_sens, 2)) * 100, \
#                                np.mean(template_match(syn_src_sens, sem_src_sens, 3)) * 100
# print("syn2sem matches: 2:{}, 3:{}".format(syn2sem_tma2, syn2sem_tma3))
# if para_sens is not None:
#     ref2sem_tma2, ref2sem_tma3 = np.mean(template_match(para_sens, sem_src_sens, 2)) * 100, \
#                                    np.mean(template_match(para_sens, sem_src_sens, 3)) * 100
#     print("ref2sem matches: 2:{}, 3:{}".format(ref2sem_tma2, ref2sem_tma3))
#     ref2syn_tma2, ref2syn_tma3 = np.mean(template_match(syn_src_sens, para_sens, 2)) * 100, \
#                                    np.mean(template_match(syn_src_sens, para_sens, 3)) * 100
#     print("ref2syn matches: 2:{}, 3:{}".format(ref2syn_tma2, ref2syn_tma3))
syn2res_tma2, syn2res_tma3 = np.mean(template_match(syn_src_sens, res_sens, 2)) * 100, \
                               np.mean(template_match(syn_src_sens, res_sens, 3)) * 100
print("syn_tma2:{}, syn_tma3:{}".format(syn2res_tma2, syn2res_tma3))
sem2res_tma2, sem2res_tma3 = np.mean(template_match(sem_src_sens, res_sens, 2)) * 100, \
                               np.mean(template_match(sem_src_sens, res_sens, 3)) * 100
print("sem_tma2:{}, sem_tma3:{}".format(sem2res_tma2, sem2res_tma3))
if para_sens is not None:
    para2res_tma2, para2res_tma3 = np.mean(template_match(para_sens, res_sens, 2)) * 100, \
                                   np.mean(template_match(para_sens, res_sens, 3)) * 100
    print("para_tma2:{}, para_tma3:{}".format(para2res_tma2, para2res_tma3))

