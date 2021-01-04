# This file is destined to wrap all the data pipelining utilities (reading, tokenizing, padding, batchifying .. )
import io
import os
import json

import torchtext.data as data
from torchtext.data import Dataset, Example
import torchtext.datasets as datasets
from torchtext.vocab import FastText, GloVe
import numpy as np
from time import time

from datasets import load_dataset
import datasets as hdatasets

# ========================================== BATCH ITERATING ENDPOINTS =================================================
VOCAB_LIMIT = 10000


class HuggingIMDB2:
    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion, sup_proportion, dev_index=1,
                 pretrained=False):
        self.data_path = os.path.join(".data", "imdb")
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>'
                                ,
                                is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, unk_token=None)
        start = time()
        try:
            train_data, test_data, unsup_data = hdatasets.Dataset.load_from_disk(self.data_path+"_train"), \
                                                hdatasets.Dataset.load_from_disk(self.data_path + "_test"), \
                                                hdatasets.Dataset.load_from_disk(self.data_path + "_unsup")
        except FileNotFoundError:
            train_data, test_data, unsup_data = load_dataset('imdb')['train'], load_dataset('imdb')['test'],\
                                                load_dataset('imdb')['unsupervised']
            train_data.save_to_disk(self.data_path+"_train")
            test_data.save_to_disk(self.data_path+"_test")
            unsup_data.save_to_disk(self.data_path+"_unsup")

        def expand_labels(datum):
            datum['label'] = [str(datum['label'])]*(max_len-1)
            return datum

        train_data, test_data = train_data.map(expand_labels), test_data.map(expand_labels)
        fields1 = {'text': text_field, 'label': label_field}
        fields2 = {'text': ('text', text_field), 'label': ('label', label_field)}
        fields3 = {'text': text_field}
        fields4 = {'text': ('text', text_field)}
        dev_start, dev_end = int(len(train_data)/5*(dev_index-1)), \
                             int(len(train_data)/5*(dev_index))
        train_start1, train_start2, train_end1, train_end2 = 0, dev_end, int(dev_start*sup_proportion),\
                                                             int(dev_end+(len(train_data)-dev_end)*sup_proportion)
        unsup_start, unsup_end = 0, int(len(unsup_data)*min(unsup_proportion, 1))
        # Since the datasets are originally sorted with the label as key, we shuffle them before reducing the supervised
        # or the unsupervised data to the first few examples. We use a fixed see to keep the same data for all
        # experiments
        np.random.seed(42)
        train_examples = [Example.fromdict(ex, fields2) for ex in train_data]
        unsup_examples = ([Example.fromdict(ex, fields4) for ex in unsup_data])
        np.random.shuffle(train_examples)
        np.random.shuffle(unsup_examples)
        train = Dataset(train_examples[train_start1:train_end1]+train_examples[train_start2:train_end2], fields1)
        val = Dataset(train_examples[dev_start:dev_end], fields1)
        test = Dataset([Example.fromdict(ex, fields2) for ex in test_data], fields1)
        unsup_train = Dataset(unsup_examples[unsup_start:unsup_end]
                              , fields3)
        vocab_dataset = Dataset(train_examples, fields1)

        unsup_test, unsup_val = test, test

        self.other_domains = self.get_other_domains(text_field, label_field, batch_size, device, max_len)
        if unsup_proportion > 1:
            for ds in [AmazonBeauty, AmazonIndus, AmazonSoftware]:
                unsup_train = Dataset(unsup_train.examples+ds(text_field, label_field, "train", max_len).examples,
                                      fields3)

        print('data loading took', time() - start)
        # build the vocabulary
        text_field.build_vocab(vocab_dataset, max_size=VOCAB_LIMIT) #, vectors="fasttext.simple.300d")
        label_field.build_vocab(train)
        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/5), device=device, shuffle=False,
            sort=False)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size/5), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        if pretrained:
            # ftxt = GloVe('6B', dim=100)
            ftxt = FastText()
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))

    def get_other_domains(self, text_field, label_field, batch_size, device, max_len):
        others = {}
        for ds in [AmazonBeauty, AmazonIndus, AmazonSoftware]:
            train, test = ds(text_field, label_field, "train", max_len), ds(text_field, label_field, "test", max_len)
            train_iter, test_iter = data.BucketIterator.splits(
                (train, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
            others[ds.name] = {'train': train_iter, 'test': test_iter}
        return others


class HuggingAGNews:
    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion, sup_proportion, dev_index=1,
                 pretrained=False):
        self.data_path = os.path.join(".data", "ag_news")
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>'
                                ,
                                is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, unk_token=None)

        start = time()

        try:
            train_data, test_data = hdatasets.Dataset.load_from_disk(self.data_path+"_train"), \
                                                hdatasets.Dataset.load_from_disk(self.data_path + "_test")
        except FileNotFoundError:
            train_data, test_data = load_dataset('ag_news')['train'], load_dataset('ag_news')['test']
            train_data.save_to_disk(self.data_path+"_train")
            test_data.save_to_disk(self.data_path+"_test")

        def expand_labels(datum):
            datum['label'] = [str(datum['label'])]*(max_len-1)
            return datum
        # lens = [len(sample['text'].split(' ')) for sample in train_data]
        # print(np.quantile(lens, [0.5, 0.7, 0.9, 0.95, 0.99]))

        train_data, test_data = train_data.map(expand_labels), test_data.map(expand_labels)
        fields1 = {'text': text_field, 'label': label_field}
        fields2 = {'text': ('text', text_field), 'label': ('label', label_field)}
        fields3 = {'text': text_field}
        fields4 = {'text': ('text', text_field)}
        len_train = 32000
        dev_start, dev_end = int(len_train/5*(dev_index-1)), \
                             int(len_train/5*(dev_index))
        train_start1, train_start2, train_end1, train_end2 = 0, dev_end, int(dev_start*sup_proportion),\
                                                             int(dev_end+(len_train-dev_end)*sup_proportion)
        unsup_start, unsup_end = len_train, int(len_train+64000*unsup_proportion)

        # Since the datasets are originally sorted with the label as key, we shuffle them before reducing the supervised
        # or the unsupervised data to the first few examples. We use a fixed see to keep the same data for all
        # experiments
        np.random.seed(42)
        train_examples = [Example.fromdict(ex, fields2) for ex in train_data]
        unsup_examples = ([Example.fromdict(ex, fields4) for ex in train_data])
        np.random.shuffle(train_examples)
        np.random.shuffle(unsup_examples)
        train = Dataset(train_examples[train_start1:train_end1]+train_examples[train_start2:train_end2], fields1)
        val = Dataset(train_examples[dev_start:dev_end], fields1)
        test = Dataset([Example.fromdict(ex, fields2) for ex in test_data], fields1)
        unsup_train = Dataset(unsup_examples[unsup_start:unsup_end]
                              , fields3)
        vocab_dataset = Dataset(train_examples, fields1)

        unsup_test, unsup_val = test, test
        self.other_domains = {}
        print('data loading took', time() - start)

        # build the vocabulary
        text_field.build_vocab(vocab_dataset, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)
        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size), device=device, shuffle=False,
            sort=False)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        if pretrained:
            # ftxt = GloVe('6B', dim=100)
            ftxt = FastText()
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class HuggingYelp:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion, sup_proportion, dev_index=1,
                 pretrained=False):
        self.data_path = os.path.join(".data", "yelp_all")
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>'
                                ,
                                is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, unk_token=None)

        start = time()
        try:
            yelp_data = hdatasets.Dataset.load_from_disk(self.data_path)
        except FileNotFoundError:
            yelp_data = load_dataset('csv', data_files={'train': os.path.join('.data', 'yelp', 'train.csv'),
                                                                    'test': os.path.join('.data', 'yelp', 'test.csv')},
                                                 column_names=['label', 'text'], version='0.0.2')
            yelp_data.save_to_disk(self.data_path)
        train_data, test_data = yelp_data['train'], yelp_data['test']

        def expand_labels(datum):
            datum['label'] = [str(datum['label'])]*(max_len-1)
            return datum
        lens = [len(sample['text'].split(' ')) for sample in train_data]

        train_data, test_data = train_data.map(expand_labels), test_data.map(expand_labels)
        fields1 = {'text': text_field, 'label': label_field}
        fields2 = {'text': ('text', text_field), 'label': ('label', label_field)}
        fields3 = {'text': text_field}
        fields4 = {'text': ('text', text_field)}

        len_train = int(len(train_data)/3)
        dev_start, dev_end = int(len_train/5*(dev_index-1)), \
                             int(len_train/5*(dev_index))
        train_start1, train_start2, train_end1, train_end2 = 0, dev_end, int(dev_start*sup_proportion),\
                                                             int(dev_end+(len_train-dev_end)*sup_proportion)
        unsup_start, unsup_end = len_train, int(len_train+len_train*2*unsup_proportion)
        # Since the datasets are originally sorted with the label as key, we shuffle them before reducing the supervised
        # or the unsupervised data to the first few examples. We use a fixed see to keep the same data for all
        # experiments
        np.random.seed(42)
        train_examples = [Example.fromdict(ex, fields2) for ex in train_data]
        unsup_examples = [Example.fromdict(ex, fields4) for ex in train_data]
        np.random.shuffle(train_examples)
        np.random.shuffle(unsup_examples)
        train = Dataset(train_examples[train_start1:train_end1]+train_examples[train_start2:train_end2], fields1)
        val = Dataset(train_examples[dev_start:dev_end], fields1)
        test = Dataset([Example.fromdict(ex, fields2) for ex in test_data], fields1)
        unsup_train = Dataset(unsup_examples[unsup_start:unsup_end], fields3)

        vocab_dataset = Dataset(train_examples, fields1)
        unsup_test, unsup_val = test, test

        self.other_domains = {}
        print('data loading took', time() - start)

        # build the vocabulary
        text_field.build_vocab(vocab_dataset, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)
        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size), device=device, shuffle=False,
            sort=False)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        if pretrained:
            ftxt = FastText()
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class IMDBData:
    def __init__(self, max_len, batch_size, max_epochs, device):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>', init_token='<go>'
                                ,
                                is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True)

        start = time()
        unsup_train, unsup_val = datasets.IMDB.splits(text_field, label_field)
        unsup_test = unsup_val
        train, val, test = unsup_train, unsup_val, unsup_test
        print('data loading took', time()-start)

        # build the vocabulary
        text_field.build_vocab(unsup_train, max_size=30000)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)
        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter, _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size / 10), device=device, shuffle=False,
            sort=False)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class UDPoSDaTA:
    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion, sup_proportion, dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=False, batch_first=True,  fix_length=max_len, pad_token='<pad>', init_token='<go>'
                                , is_target=True)#init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len-1, batch_first=True)

        # make splits for data
        #unsup_train, unsup_val, unsup_test = MyPennTreebank.splits(text_field)
        #unsup_train, unsup_val, unsup_test = datasets.PennTreebank.splits(text_field)
        #unsup_train, unsup_val, unsup_test = datasets.WikiText2.splits(text_field)
        unsup_train, unsup_val, unsup_test = UDPOS1_2.splits((('text', text_field), ('label', label_field)))
        #unsup_train, unsup_val, unsup_test = YahooLM.splits(text_field)
        train, val, test = UDPOS1_2.splits((('text', text_field), ('label', label_field)))

        # build the vocabulary
        text_field.build_vocab(unsup_train)#, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)
        # self.train_iter, _,  _ = data.BPTTIterator.splits((unsup_train, unsup_val, unsup_test),
        #                                                                     batch_size=batch_size, bptt_len=max_len,
        #                                                                     device=device, repeat=False, shuffle=False,
        #                                                                     sort=False)
        # _, self.unsup_val_iter,  _ = data.BPTTIterator.splits((unsup_train, unsup_val, unsup_test),
        #                                                                     batch_size=int(batch_size/10), bptt_len=max_len,
        #                                                                     device=device, repeat=False, shuffle=False,
        #                                                                     sort=False)
        # Remaking splits according to supervision proportions
        exlist = [ex for ex in train+val]
        train =Dataset(exlist, {'text': text_field, 'label': label_field})
        dev_start, dev_end = int(len(train) / 5 * (dev_index - 1)), \
                             int(len(train) / 5 * (dev_index))
        train_start1, train_start2, train_end1, train_end2 = 0, dev_end, int(dev_start * sup_proportion), \
                                                             int(dev_end + (len(train) - dev_end) * sup_proportion)
        unsup_start, unsup_end = 0, int(len(unsup_train) * unsup_proportion)
        val = Dataset(train[dev_start:dev_end], {'text': text_field, 'label': label_field})
        train = Dataset(train[train_start1:train_end1] + train[train_start2:train_end2],
                        {'text': text_field, 'label': label_field})
        unsup_train = Dataset(unsup_train[unsup_start:unsup_end], {'text': text_field})
        self.other_domains = {}

        # make iterator for splits

        self.train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/10), device=device, shuffle=False, sort=False)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        if pretrained:
            ftxt = GloVe('6B', dim=100)
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))

class UDPoSDaTAFull:
    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion, sup_proportion, dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=True, batch_first=True,  fix_length=max_len, pad_token='<pad>', init_token='<go>'
                                , is_target=True)#init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len-1, batch_first=True)

        # make splits for data
        #unsup_train, unsup_val, unsup_test = MyPennTreebank.splits(text_field)
        #unsup_train, unsup_val, unsup_test = datasets.PennTreebank.splits(text_field)
        #unsup_train, unsup_val, unsup_test = datasets.WikiText2.splits(text_field)
        unsup_train, unsup_val, unsup_test = UDPOS1_2.splits((('text', text_field), ('label', label_field)))
        #unsup_train, unsup_val, unsup_test = YahooLM.splits(text_field)
        train, val, test = UDPOS1_2.splits((('text', text_field), ('label', label_field)))

        # build the vocabulary
        text_field.build_vocab(unsup_train)#, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits

        self.train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/10), device=device, shuffle=False, sort=False)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        if pretrained:
            ftxt = FastText()
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class NLIGenData2:
    def __init__(self, max_len, batch_size, max_epochs, device, pretrained):
        text_field = data.Field(lower=True, batch_first=True,  fix_length=max_len, init_token='<go>', eos_token='<eos>',
                                unk_token='<unk>', pad_token='<pad>')
        label_field = data.Field(fix_length=max_len-1, batch_first=True)

        # make splits for data
        unsup_train, unsup_val, unsup_test = NLIGen.splits(text_field)
        train, val, test = datasets.UDPOS.splits((('text', text_field), ('label', label_field)))

        # build the vocabulary
        text_field.build_vocab(unsup_train)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits
        self.train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/10), device=device, shuffle=True, sort=False)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, unsup_val, unsup_test), batch_size=int(batch_size), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        if pretrained:
            ftxt = FastText()
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class Wiki2Data:
    def __init__(self, max_len, batch_size, max_epochs, device):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, init_token='<go>',
                                eos_token='<eos>',)

        # make splits for data
        train, val, test = MyWikiText2.splits(text_field)

        # build the vocabulary
        text_field.build_vocab(train)  # , vectors="fasttext.simple.300d")

        # make iterator for splits
        self.train_iter, self.val_iter,  self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=False)
        self.val_iter.shuffle = False
        self.test_iter.shuffle = False

        self.vocab = text_field.vocab
        self.tags = None
        self.text_field = text_field
        self.label_field = None
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class NLIGenData:
    def __init__(self, max_len, batch_size, max_epochs, device):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, init_token='<go>',
                                eos_token='<eos>',)

        # make splits for data
        train, val, test = NLIGen.splits(text_field)

        # build the vocabulary
        text_field.build_vocab(train)  # , vectors="fasttext.simple.300d")

        # make iterator for splits
        self.train_iter, self.val_iter,  self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=False)
        self.val_iter.shuffle = False
        self.test_iter.shuffle = False

        self.vocab = text_field.vocab
        self.tags = None
        self.text_field = text_field
        self.label_field = None
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


class OntoGenData:
    def __init__(self, max_len, batch_size, max_epochs, device, pretrained):
        text_field = data.Field(lower=True, batch_first=True,  fix_length=max_len, init_token='<go>', eos_token='<eos>',
                                unk_token='<unk>', pad_token='<pad>')
        label_field = data.Field(fix_length=max_len-1, batch_first=True)

        # make splits for data
        unsup_train, unsup_val, unsup_test = OntoGen.splits([('text', text_field)])

        # build the vocabulary
        text_field.build_vocab(unsup_train, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(unsup_train)

        # make iterator for splits
        self.train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.unsup_val_iter,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/10), device=device, shuffle=True, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        if pretrained:
            ftxt = FastText()
            self.wvs = ftxt.get_vecs_by_tokens(self.vocab.itos)
        else:
            self.wvs = None

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                self.train_iter.init_epoch()
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            self.val_iter.init_epoch()
        elif split == 'test':
            self.test_iter.init_epoch()
        elif split == 'unsup_valid':
            self.unsup_val_iter.init_epoch()
        else:
            raise NameError('Misspelled split name : {}'.format(split))


# ======================================================================================================================
# ========================================== OTHER UTILITIES ===========================================================


class MyVocab:
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi


class LanguageModelingDataset(data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, text_field, newline_eos=True,
                 encoding='utf-8', **kwargs):
        """Create a LanguageModelingDataset given a path and a field.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field)]
        examples = []
        seq_lens = []
        with io.open(path, encoding=encoding) as f:
            for i, line in enumerate(f):
                processed_line = text_field.preprocess(line)
                for sentence in ' '.join(processed_line).replace('! ', '<spl>')\
                       .replace('? ', '<spl>').replace('. ', '<spl>').split('<spl>'):
                   if len(sentence) > 1 and '=' not in sentence:
                       examples.append(data.Example.fromlist([(sentence+'.').split(' ')], fields))
                       seq_lens.append(len(sentence.split(' ')))
                # if len(processed_line) > 1 and not any(['=' in tok for tok in  processed_line]):
                #     examples.append(data.Example.fromlist([processed_line], fields))
            print("Mean length: ", sum(seq_lens)/len(seq_lens), ' Quantiles .25, 0.5, 0.7, and 0.9 :',
                  np.quantile(seq_lens, [0.25, 0.5, 0.7, 0.9, 0.95, 0.99]), 'std:', np.std(seq_lens),
                  'n_examples:', len(seq_lens))

        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)


class MyPennTreebank(LanguageModelingDataset):
    """The Penn Treebank dataset.
    A relatively small dataset originally created for POS tagging.

    References
    ----------
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank
    """

    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    name = 'penn-treebank'
    dirname = ''

    @classmethod
    def splits(cls, text_field, root='.data', train='ptb.train.txt',
               validation='ptb.valid.txt', test='ptb.test.txt',
               **kwargs):
        """Create dataset objects for splits of the Penn Treebank dataset.

        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory where the data files will be stored.
            train: The filename of the train data. Default: 'ptb.train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'ptb.valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'ptb.test.txt'.
        """
        return super(MyPennTreebank, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        """Create iterator objects for splits of the Penn Treebank dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory where the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)

class YahooLM(LanguageModelingDataset):
    """The Penn Treebank dataset.
    A relatively small dataset originally created for POS tagging.

    References
    ----------
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank
    """

    urls = []
    name = 'yahoo'
    dirname = ''

    @classmethod
    def splits(cls, text_field, root='.data', train='train.txt',
               validation='dev.txt', test='dev.txt',
               **kwargs):
        """Create dataset objects for splits of the Penn Treebank dataset.

        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory where the data files will be stored.
            train: The filename of the train data. Default: 'ptb.train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'ptb.valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'ptb.test.txt'.
        """
        return super(YahooLM, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        """Create iterator objects for splits of the Penn Treebank dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory where the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)


class MyWikiText2(LanguageModelingDataset):

    urls = ['https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip']
    name = 'wikitext-2'
    dirname = 'wikitext-2'

    @classmethod
    def splits(cls, text_field, root='.data', train='wiki.train.tokens',
               validation='wiki.valid.tokens', test='wiki.test.tokens',
               **kwargs):
        """Create dataset objects for splits of the WikiText-2 dataset.

        This is the most flexible way to use the dataset.

        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'wiki.train.tokens'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'wiki.valid.tokens'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'wiki.test.tokens'.
        """
        return super(MyWikiText2, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)


class NLIGen(LanguageModelingDataset):

    urls = ['https://raw.githubusercontent.com/schmiflo/crf-generation/master/generated-text/train']
    name = 'nli_gen'
    dirname = 'nli_gen'

    @classmethod
    def splits(cls, text_field, root='.data', train='train.txt',
               validation='test.txt', test='test.txt',
               **kwargs):
        """Create dataset objects for splits of the WikiText-2 dataset.

        This is the most flexible way to use the dataset.

        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'wiki.train.tokens'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'wiki.valid.tokens'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'wiki.test.tokens'.
        """
        return super(NLIGen, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)


class UDPOS1_2(Dataset):
    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    urls = []
    dirname = 'en-ud-v1'
    name = 'udpos'

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        columns = []

        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    elements = list(line.split(separator))
                    for i, column in enumerate([elements[1], elements[3]]):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(UDPOS1_2, self).__init__(examples, fields, **kwargs)
    @classmethod
    def splits(cls, fields, root=".data", train="en-ud-train.conllu",
               validation="en-ud-dev.conllu",
               test="en-ud-test.conllu", **kwargs):
        """Loads the Universal Dependencies Version 1 POS Tagged
        data.
        """

        return super(UDPOS1_2, cls).splits(
            fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)


class OntoGen(Dataset):
    urls = []
    dirname = 'ontonotes'
    name = 'ontonotes'

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        columns = []

        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns and 0 < len(columns[0]) <= 16:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    elements = list(line.split(separator))
                    for i, column in enumerate([elements[0]]):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        print("Collected {} examples from {}".format(len(examples), path))
        super(OntoGen, self).__init__(examples, fields, **kwargs)
    @classmethod
    def splits(cls, fields, root=".data", train="onto.train.ner",
               validation="onto.development.ner",
               test="onto.test.ner", **kwargs):
        """Loads the Universal Dependencies Version 1 POS Tagged
        data.
        """
        print("Loading Ontonotes data ...")
        return super(OntoGen, cls).splits(
            path=os.path.join(".data", "ontonotes"), fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)


class AmazonBase(Dataset):

    def __init__(self, text_field, label_field, split, max_len, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        pos_examples = []
        neg_examples = []
        label_list = []

        # Counting the maximum number from each class
        with open(self.data_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                json_line = json.loads(line)
                if "reviewText" in json_line:
                    review, stars = json_line["reviewText"].replace('\n', ' ').strip(), json_line["overall"]
                    # if stars != 3:
                    label = int(stars > 3)
                    label_list.append(label)
        max_per_class = min(sum(label_list), len(label_list) - sum(label_list))
        label_list = []
        with open(self.data_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                json_line = json.loads(line)
                if "reviewText" in json_line:
                    review, stars = json_line["reviewText"].replace('\n', ' ').strip(), json_line["overall"]
                    # if stars != 3:
                    label = int(stars > 3)
                    # Checking that the number of examples from this class doesn't break the balance
                    if label and sum(label_list) < max_per_class:
                        label_list.append(label)
                        pos_examples.append(data.Example.fromlist([review, [str(label)] * (max_len - 1)], fields))
                    elif not label and len(label_list) - sum(label_list) < max_per_class:
                        label_list.append(label)
                        neg_examples.append(data.Example.fromlist([review, [str(label)] * (max_len - 1)], fields))
        np.random.seed(42)
        if split == "train":
            pos_examples = pos_examples[0:int(len(pos_examples) / 2)]
            neg_examples = neg_examples[0:int(len(neg_examples) / 2)]
            examples = pos_examples+neg_examples
            np.random.shuffle(examples)
            label_list = label_list[0:int(len(label_list) / 2)]
        else:
            pos_examples = pos_examples[int(len(pos_examples) / 2):len(pos_examples)]
            neg_examples = neg_examples[int(len(neg_examples) / 2):len(neg_examples)]
            examples = pos_examples+neg_examples
            np.random.shuffle(examples)
            label_list = label_list[int(len(label_list) / 2):len(label_list)]
        print("{}'s {} split contains {} balanced examples".format(self.name, split, len(label_list)))

        super(AmazonBase, self).__init__(examples, fields, **kwargs)


class AmazonBeauty(AmazonBase):
    name = "amazon_beauty"
    data_path = os.path.join(".data", "amazon", "Luxury_Beauty_5.json")


class AmazonSoftware(AmazonBase):
    name = "amazon_software"
    data_path = os.path.join(".data", "amazon", "Software_5.json")


class AmazonIndus(AmazonBase):
    name = "amazon_industrial"
    data_path = os.path.join(".data", "amazon", "Industrial_and_Scientific_5.json")