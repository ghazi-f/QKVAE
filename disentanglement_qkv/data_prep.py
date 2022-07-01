# This file is destined to wrap all the data pipelining utilities (reading, tokenizing, padding, batchifying .. )
import io
import os
import json
import re

import torchtext.data as data
from torchtext.data import Dataset, Example
import torchtext.datasets as datasets
from torchtext.vocab import FastText, GloVe
import numpy as np
import torch
from time import time
from components.links import LOCAL_ONLY

from datasets import load_dataset
from tokenizers.models import BPE, WordLevel
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import BartTokenizerFast, BarthezTokenizerFast
import datasets as hdatasets

# ========================================== BATCH ITERATING ENDPOINTS =================================================
VOCAB_LIMIT = 10000
TEST_DIVIDE_BS = 8


class BARTParaNMT:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):

        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        self.divide_bs = TEST_DIVIDE_BS

        np.random.seed(42)
        # Loading Data
        folder = os.path.join(".data", "paranmt")
        train_path, valid_path, test_path = os.path.join(folder, 'train.txt'), os.path.join(folder, 'dev.txt'), \
                                            os.path.join(folder, 'test.txt')
        self.dataset = load_dataset('csv', data_files={'train': train_path, 'valid': valid_path, 'test': test_path},
                                    delimiter='\t', column_names=['text', 'para'], keep_in_memory=True)

        # print("Original text Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['text'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Original text length:", np.mean(tr_len))
        # print("Paraphrase Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['para'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Paraphrase length:", np.mean(tr_len))
        self.dataset = {'train': self.dataset['train'][:],
                        'valid': self.dataset['valid'][:],
                        'test': self.dataset['test'][:]}

        # Getting Tokenizer
        self.tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base', local_files_only=LOCAL_ONLY)

        # Getting vocabulary object
        stoi = self.tokenizer.get_vocab()
        itos = [None] * len(stoi)
        for k, v in stoi.items():
            itos[v] = k
        self.vocab = MyVocab(itos, stoi)

        # Setting up iterators
        self.shuffle_split('train'), self.shuffle_split('valid'), self.shuffle_split('test')
        self.train_iter, self.enc_train_iter = MyBARTIter(self, self.dataset['train']), \
                                               MyBARTIter(self, self.dataset['train'])
        self.val_iter, self.test_iter = MyBARTIter(self, self.dataset['valid'], divide_bs=self.divide_bs), \
                                        MyBARTIter(self, self.dataset['test'], divide_bs=self.divide_bs)

        self.tags, self.wvs = None, None

    def redefine_max_len(self, new_len):
        folder = os.path.join(".data", "paranmt")
        train_path, valid_path, test_path = os.path.join(folder, 'train.txt'), os.path.join(folder, 'dev.txt'), \
                                            os.path.join(folder, 'test.txt')
        print("Reloading the data with max words per sentence : ", new_len)
        self.dataset = load_dataset('csv', data_files={'train': train_path, 'valid': valid_path, 'test': test_path},
                                    delimiter='\t', column_names=['text', 'para'], keep_in_memory=True)
        self.dataset = self.dataset.filter(lambda x: len(x['text'].split()) < new_len)
        self.dataset = {'train': self.dataset['train'][:],
                        'valid': self.dataset['valid'][:],
                        'test': self.dataset['test'][:]}
        self.reinit_iterator('train')
        self.reinit_iterator('valid')
        self.reinit_iterator('test')

    def shuffle_split(self, split):
        assert split in ('train', 'valid', 'test')
        rrange = np.arange(len(self.dataset[split]['text']))
        np.random.shuffle(rrange)
        self.dataset[split]['text'] = np.array(self.dataset[split]['text'])[rrange].tolist()
        self.dataset[split]['para'] = np.array(self.dataset[split]['para'])[rrange].tolist()

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.shuffle_split('train')
                self.train_iter = MyBARTIter(self, self.dataset['train'])
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.shuffle_split('valid')
            self.val_iter = MyBARTIter(self, self.dataset['valid'], divide_bs=self.divide_bs)
        elif split == 'test':
            self.shuffle_split('test')
            self.test_iter = MyBARTIter(self, self.dataset['test'], divide_bs=self.divide_bs)
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class BARTFrSbt:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):

        """
        quantiles:  [ 5.  7. 11. 14. 22.]
        len of files dev.txt: 6.0464+/-4.364796334309312"""
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        self.divide_bs = TEST_DIVIDE_BS

        np.random.seed(42)
        # Loading Data
        folder = os.path.join(".data", "fr_sbt")
        train_path, valid_path, test_path = os.path.join(folder, 'train.txt'), os.path.join(folder, 'dev.txt'), \
                                            os.path.join(folder, 'test.txt')
        self.dataset = load_dataset('csv', data_files={'train': train_path, 'valid': valid_path, 'test': test_path},
                                    delimiter='\t', column_names=['text', 'para'], keep_in_memory=True)

        # print("Original text Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['text'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Original text length:", np.mean(tr_len))
        # print("Paraphrase Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['para'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Paraphrase length:", np.mean(tr_len))
        self.dataset = {'train': self.dataset['train'][:],
                        'valid': self.dataset['valid'][:],
                        'test': self.dataset['test'][:]}
        # Getting Tokenizer
        self.tokenizer = BarthezTokenizerFast.from_pretrained('moussaKam/barthez', local_files_only=LOCAL_ONLY)

        # Getting vocabulary object
        stoi = self.tokenizer.get_vocab()
        itos = [None] * len(stoi)
        for k, v in stoi.items():
            itos[v] = k
        self.vocab = MyVocab(itos, stoi)

        # Setting up iterators
        self.shuffle_split('train'), self.shuffle_split('valid'), self.shuffle_split('test')
        self.train_iter, self.enc_train_iter = MyBARTIter(self, self.dataset['train']), \
                                               MyBARTIter(self, self.dataset['train'])
        self.val_iter, self.test_iter = MyBARTIter(self, self.dataset['valid'], divide_bs=self.divide_bs), \
                                        MyBARTIter(self, self.dataset['test'], divide_bs=self.divide_bs)

        self.tags, self.wvs = None, None

    def redefine_max_len(self, new_len):
        folder = os.path.join(".data", "fr_sbt")
        train_path, valid_path, test_path = os.path.join(folder, 'train.txt'), os.path.join(folder, 'dev.txt'), \
                                            os.path.join(folder, 'test.txt')
        print("Reloading the data with max words per sentence : ", new_len)
        self.dataset = load_dataset('csv', data_files={'train': train_path, 'valid': valid_path, 'test': test_path},
                                    delimiter='\t', column_names=['text', 'para'], keep_in_memory=True)
        self.dataset = self.dataset.filter(lambda x: len(x['text'].split()) < new_len)
        self.dataset = {'train': self.dataset['train'][:],
                        'valid': self.dataset['valid'][:],
                        'test': self.dataset['test'][:]}
        self.reinit_iterator('train')
        self.reinit_iterator('valid')
        self.reinit_iterator('test')

    def shuffle_split(self, split):
        assert split in ('train', 'valid', 'test')
        rrange = np.arange(len(self.dataset[split]['text']))
        np.random.shuffle(rrange)
        self.dataset[split]['text'] = np.array(self.dataset[split]['text'], dtype=object)[rrange].tolist()
        self.dataset[split]['para'] = np.array(self.dataset[split]['para'], dtype=object)[rrange].tolist()

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.shuffle_split('train')
                self.train_iter = MyBARTIter(self, self.dataset['train'])
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.shuffle_split('valid')
            self.val_iter = MyBARTIter(self, self.dataset['valid'], divide_bs=self.divide_bs)
        elif split == 'test':
            self.shuffle_split('test')
            self.test_iter = MyBARTIter(self, self.dataset['test'], divide_bs=self.divide_bs)
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class ParaNMTCuratedData:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):

        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        self.divide_bs = 4

        np.random.seed(42)
        # Loading Data
        folder = os.path.join(".data", "paranmt")
        train_path, valid_path, test_path = os.path.join(folder, 'train.txt'), os.path.join(folder, 'dev.txt'), \
                                os.path.join(folder, 'test.txt')
        self.dataset = load_dataset('csv', data_files={'train': train_path, 'valid': valid_path, 'test': test_path},
                                    delimiter='\t', column_names=['text', 'para'], keep_in_memory=True)

        # print("Original text Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['text'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Original text length:", np.mean(tr_len))
        # print("Paraphrase Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['para'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Paraphrase length:", np.mean(tr_len))
        self.dataset = {'train': self.dataset['train'][:],
                        'valid': self.dataset['valid'][:],
                        'test': self.dataset['test'][:]}

        # Getting Tokenizer
        is_bpe = False
        if is_bpe:
            tok_name, tok_model, tok_trainer = "paranmttest.json", BPE, BpeTrainer
        else:
            tok_name, tok_model, tok_trainer = "paranmttest_wl.json", WordLevel, WordLevelTrainer
        tokenizer_path = os.path.join("tokenizers", tok_name)
        if not os.path.exists(tokenizer_path):
            # Initialize a tokenizer
            trainer = tok_trainer(vocab_size=VOCAB_LIMIT, min_frequency=2,
                                 special_tokens=['<unk>', '<eos>', '<go>', '<pad>'], show_progress=True,
                                 continuing_subword_prefix='ğ')
            self.tokenizer = Tokenizer(tok_model(unk_token='<unk>'))
            self.tokenizer.pre_tokenizer = Whitespace()
            # Customize training
            self.tokenizer.train(files=[train_path], trainer=trainer)

            # # Save files to disk
            if not os.path.exists("tokenizers"):
                os.mkdir("tokenizers")
            self.tokenizer.save(tokenizer_path)
        else:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=max_len)

        # Getting vocabulary object
        stoi = self.tokenizer.get_vocab()
        itos = [None] * len(stoi)
        for k, v in stoi.items():
            itos[v] = k
        self.vocab = MyVocab(itos, stoi)

        # Setting up iterators
        self.shuffle_split('train'), self.shuffle_split('valid'), self.shuffle_split('test')
        self.train_iter, self.enc_train_iter = MyIter(self, self.dataset['train']), \
                                               MyIter(self, self.dataset['train'])
        self.val_iter, self.test_iter = MyIter(self, self.dataset['valid'], divide_bs=self.divide_bs), \
                                        MyIter(self, self.dataset['test'], divide_bs=self.divide_bs)

        self.tags = None
        if pretrained:
            ftxt = FastText()
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def redefine_max_len(self, new_len):
        folder = os.path.join(".data", "paranmt")
        train_path, valid_path, test_path = os.path.join(folder, 'train.txt'), os.path.join(folder, 'dev.txt'), \
                                os.path.join(folder, 'test.txt')
        print("Reloading the data with max words per sentence : ", new_len)
        self.dataset = load_dataset('csv', data_files={'train': train_path, 'valid': valid_path, 'test': test_path},
                                    delimiter='\t', column_names=['text', 'para'], keep_in_memory=True)
        self.dataset = self.dataset.filter(lambda x: len(x['text'].split()) < new_len)
        self.dataset = {'train': self.dataset['train'][:],
                        'valid': self.dataset['valid'][:],
                        'test': self.dataset['test'][:]}
        self.reinit_iterator('train')
        self.reinit_iterator('valid')
        self.reinit_iterator('test')

    def shuffle_split(self, split):
        assert split in ('train', 'valid', 'test')
        rrange = np.arange(len(self.dataset[split]['text']))
        np.random.shuffle(rrange)
        self.dataset[split]['text'] = np.array(self.dataset[split]['text'])[rrange].tolist()
        self.dataset[split]['para'] = np.array(self.dataset[split]['para'])[rrange].tolist()

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.shuffle_split('train')
                self.train_iter = MyIter(self, self.dataset['train'])
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.shuffle_split('valid')
            self.val_iter = MyIter(self, self.dataset['valid'], divide_bs=self.divide_bs)
        elif split == 'test':
            self.shuffle_split('test')
            self.test_iter = MyIter(self, self.dataset['test'], divide_bs=self.divide_bs)
        else:
            raise NameError('Misspelled split name : {}'.format(split))


# ======================================================================================================================
# ========================================== OTHER UTILITIES ===========================================================


class MyVocab:
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi


class ParaExample:
    def __init__(self, text, para):
        self.text = text
        self.para = para


class MyIter:
    def __init__(self, data_obj, dat, divide_bs=1):
        self.data_obj = data_obj
        self.dat = dat
        self.divide_bs = divide_bs

    def __iter__(self):
        this_bs, device = int(self.data_obj.batch_size/self.divide_bs), self.data_obj.device
        for i in range(0, len(self.dat["text"]), this_bs):
            text = torch.Tensor([self.data_obj.tokenizer.encode('<go> '+ex+' <pad>'*self.data_obj.max_len).ids
                                 for ex in self.dat["text"][i: i + this_bs]]).long().to(device)
            para = torch.Tensor([self.data_obj.tokenizer.encode('<go> '+ex+' <pad>'*self.data_obj.max_len).ids
                                 for ex in self.dat["para"][i: i + this_bs]]).long().to(device)
            yield ParaExample(text, para)

    def __len__(self):
        return int(len(self.dat['text'])/(self.data_obj.batch_size/self.divide_bs))


class MyBARTIter:
    def __init__(self, data_obj, dat, divide_bs=1):
        self.data_obj = data_obj
        self.dat = dat
        self.divide_bs = divide_bs

    def __iter__(self):
        this_bs, device = int(self.data_obj.batch_size/self.divide_bs), self.data_obj.device
        for i in range(0, len(self.dat["text"]), this_bs):
            text = self.data_obj.tokenizer(self.dat["text"][i: i + this_bs], max_length=self.data_obj.max_len,
                                           truncation=True, padding="max_length",
                                           return_tensors='pt')["input_ids"].to(device)
            para = None
            yield ParaExample(text, para)

    def __len__(self):
        return int(len(self.dat['text'])/(self.data_obj.batch_size/self.divide_bs))
