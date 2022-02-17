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
from time import time

from datasets import load_dataset
import datasets as hdatasets

# ========================================== BATCH ITERATING ENDPOINTS =================================================
VOCAB_LIMIT = 10000


class HuggingYelp:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):
        self.data_path = os.path.join(".data", "yelp_all")
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>'
                                ,
                                is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, unk_token=None)

        start = time()
        try:
            train_data = hdatasets.Dataset.load_from_disk(os.path.join(self.data_path, 'train'))
            test_data = hdatasets.Dataset.load_from_disk(os.path.join(self.data_path, 'test'))
        except FileNotFoundError as e:
            print("Proceeding to read datasets for the first time because of error:\n", e)
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
        unsup_start, unsup_end = 0, 100000
        # Since the datasets are originally sorted with the label as key, we shuffle them before reducing the supervised
        # or the unsupervised data to the first few examples. We use a fixed see to keep the same data for all
        # experiments
        np.random.seed(42)
        train_examples = [Example.fromdict(ex, fields2) for i, ex in enumerate(train_data) if i<unsup_end]
        unsup_examples = [Example.fromdict(ex, fields4) for i, ex in enumerate(train_data) if i<unsup_end]
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
        label_field.build_vocab(vocab_dataset)

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


class HuggingYelp2:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>' ,is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, unk_token=None)

        start = time()

        train, val, test = BinaryYelp.splits((('text', text_field), ('label', label_field)))

        # np.random.shuffle(train_examples)
        fields1 = {'text': text_field, 'label': label_field}
        train = Dataset(train, fields1)
        val = Dataset(val, fields1)
        test = Dataset(test, fields1)
        print('data loading took', time() - start)

        # build the vocabulary
        text_field.build_vocab(train, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.enc_train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)

        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size/10), device=device, shuffle=False, sort=False)

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

class SupYelpData:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>' ,is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, pad_token="o")

        start = time()

        train, val, test = SupYelp.splits((('text', text_field), ('label', label_field)))

        # np.random.shuffle(train_examples)
        fields1 = {'text': text_field, 'label': label_field}
        train = Dataset(train, fields1)
        val = Dataset(val, fields1)
        test = Dataset(test, fields1)
        print('data loading took', time() - start)

        # build the vocabulary
        text_field.build_vocab(train, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.enc_train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)

        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size/10), device=device, shuffle=False, sort=False)

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

class SupNLIData:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>' ,is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, pad_token="o")

        start = time()

        train, val, test = SupNLI.splits((('text', text_field), ('label', label_field)))

        # np.random.shuffle(train_examples)
        fields1 = {'text': text_field, 'label': label_field}
        train = Dataset(train, fields1)
        val = Dataset(val, fields1)
        test = Dataset(test, fields1)
        print('data loading took', time() - start)

        # build the vocabulary
        text_field.build_vocab(train, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.enc_train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)

        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size/10), device=device, shuffle=False, sort=False)

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


class HuggingYelpReg:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>' ,is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')

        start = time()

        train, val, test = RegYelp.splits((('text', text_field),))

        # np.random.shuffle(train_examples)
        fields1 = {'text': text_field}
        train = Dataset(train, fields1)
        val = Dataset(val, fields1)
        test = Dataset(test, fields1)
        print('data loading took', time() - start)

        # build the vocabulary
        text_field.build_vocab(train, max_size=VOCAB_LIMIT)  # , vectors="fasttext.simple.300d")

        # make iterator for splits
        self.train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.enc_train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)

        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size/10), device=device, shuffle=False, sort=False)

        self.vocab = text_field.vocab
        self.tags = None
        self.text_field = text_field
        self.label_field = None
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



class GermanNLIGenData2:
    def __init__(self, max_len, batch_size, max_epochs, device, pretrained):
        text_field = data.Field(lower=True, batch_first=True,  fix_length=max_len, init_token='<go>', eos_token='<eos>',
                                unk_token='<unk>', pad_token='<pad>')

        # make splits for data
        unsup_train, unsup_val, unsup_test = GermanNLIGen.splits(text_field)

        # build the vocabulary
        text_field.build_vocab(unsup_train)

        # make iterator for splits
        self.train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.enc_train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.val_iter,  self.test_iter = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/10), device=device, shuffle=True, sort=False)

        self.vocab = text_field.vocab
        self.tags = None
        self.text_field = text_field
        self.label_field = None
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

        # make splits for data
        unsup_train, unsup_val, unsup_test = NLIGen.splits(text_field)

        # build the vocabulary
        text_field.build_vocab(unsup_train)

        # make iterator for splits
        self.train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.enc_train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.val_iter,  self.test_iter = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/10), device=device, shuffle=True, sort=False)

        self.vocab = text_field.vocab
        self.tags = None
        self.text_field = text_field
        self.label_field = None
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


class Wiki21GenData:
    def __init__(self, max_len, batch_size, max_epochs, device, pretrained):
        text_field = data.Field(lower=True, batch_first=True,  fix_length=max_len, init_token='<go>', eos_token='<eos>',
                                unk_token='<unk>', pad_token='<pad>')

        # make splits for data
        unsup_train, unsup_val, unsup_test = WikiGen.splits(text_field)

        # build the vocabulary
        text_field.build_vocab(unsup_train)

        # make iterator for splits
        self.train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.enc_train_iter, _,  _ = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        _, self.val_iter,  self.test_iter = data.BucketIterator.splits(
            (unsup_train, unsup_val, unsup_test), batch_size=int(batch_size/10), device=device, shuffle=True, sort=False)

        self.vocab = text_field.vocab
        self.tags = None
        self.text_field = text_field
        self.label_field = None
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
        self.enc_train_iter, _,  _ = data.BucketIterator.splits(
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


class NLIGen(LanguageModelingDataset):

    urls = ['https://raw.githubusercontent.com/schmiflo/crf-generation/master/generated-text/train']
    name = 'nli_gen'
    dirname = 'nli_gen'

    @classmethod
    def splits(cls, text_field, root='.data', train='train.txt',
               validation='valid.txt', test='test.txt',
               **kwargs):
        return super(NLIGen, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)


class WikiGen(LanguageModelingDataset):

    urls = []
    name = 'wiki21'
    dirname = 'wiki21'

    @classmethod
    def splits(cls, text_field, root='.data', train='train.txt',
               validation='dev.txt', test='test.txt',
               **kwargs):
        return super(WikiGen, cls).splits(path=os.path.join(root, 'wiki21'),
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)

class GermanNLIGen(LanguageModelingDataset):

    urls = ['https://raw.githubusercontent.com/schmiflo/crf-generation/master/generated-text/train']
    name = 'de_nli'
    dirname = 'de_nli'

    @classmethod
    def splits(cls, text_field, root='.data', train='de_train.tsv',
               validation='de_dev.tsv', test='de_test.tsv',
               **kwargs):
        return super(GermanNLIGen, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='.data',
              vectors=None, **kwargs):
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)

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


class BinaryYelp(Dataset):
    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    urls = []
    dirname = ''
    name = ''

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", verbose=1, shuffle_seed=42, **kwargs):
        examples = []
        n_examples, n_words, n_chars = 0, [], []
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                sen, lab = line.split('\t')
                sen, lab = sen.split(), [int(lab)] * len(list(sen.split()))
                examples.append(data.Example.fromlist([sen, lab], fields))
                n_examples += 1
                n_words.append(len(sen))
        if verbose:
            print("Dataset has {}  examples. statistics:\n -words: {}+-{}(quantiles(0.5, 0.7, 0.9, 0.95, "
                  "0.99:{},{},{},{},{})".format(n_examples, np.mean(n_words), np.std(n_words),
                                                *np.quantile(n_words, [0.5, 0.7, 0.9, 0.95, 0.99])))
        np.random.seed(42)
        np.random.shuffle(examples)
        super(BinaryYelp, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root=".data", train="yelp.train.tsv",
               validation="yelp.dev.tsv",
               test="yelp.test.tsv", **kwargs):
        """Loads the Universal Dependencies Version 1 POS Tagged
        data.
        """

        return super(BinaryYelp, cls).splits(
            path=os.path.join(".data", "binary_yelp"), fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)

class SupYelp(Dataset):
    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    urls = []
    dirname = ''
    name = ''

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", verbose=1, shuffle_seed=42, **kwargs):
        examples = []
        n_examples, n_words, n_chars = 0, [], []
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                sen, lab = line.strip().split('\t')
                sen, lab = sen.split(), lab.split()
                examples.append(data.Example.fromlist([sen, lab], fields))
                n_examples += 1
                n_words.append(len(sen))
        if verbose:
            print("Dataset has {}  examples. statistics:\n -words: {}+-{}(quantiles(0.5, 0.7, 0.9, 0.95, "
                  "0.99:{},{},{},{},{})".format(n_examples, np.mean(n_words), np.std(n_words),
                                                *np.quantile(n_words, [0.5, 0.7, 0.9, 0.95, 0.99])))
        np.random.seed(42)
        np.random.shuffle(examples)
        super(SupYelp, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root=".data", train="yelp.train_labeled.txt",
               validation="yelp.dev_labeled.txt",
               test="yelp.test_labeled.txt", **kwargs):
        """Loads the Universal Dependencies Version 1 POS Tagged
        data.
        """

        return super(SupYelp, cls).splits(
            path=os.path.join(".data", "binary_yelp"), fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)

class SupNLI(Dataset):
    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    urls = []
    dirname = ''
    name = ''

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", verbose=1, shuffle_seed=42, **kwargs):
        examples = []
        n_examples, n_words, n_chars = 0, [], []
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                sen, lab = line.strip().split('\t')
                sen, lab = sen.split(), lab.split()
                examples.append(data.Example.fromlist([sen, lab], fields))
                n_examples += 1
                n_words.append(len(sen))
        if verbose:
            print("Dataset has {}  examples. statistics:\n -words: {}+-{}(quantiles(0.5, 0.7, 0.9, 0.95, "
                  "0.99:{},{},{},{},{})".format(n_examples, np.mean(n_words), np.std(n_words),
                                                *np.quantile(n_words, [0.5, 0.7, 0.9, 0.95, 0.99])))
        np.random.seed(42)
        np.random.shuffle(examples)
        super(SupNLI, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root=".data", train="train_labeled.txt",
               validation="valid_labeled.txt",
               test="test_labeled.txt", **kwargs):
        """Loads the Universal Dependencies Version 1 POS Tagged
        data.
        """

        return super(SupNLI, cls).splits(
            path=os.path.join(".data", "nli_gen", "nli_gen"), fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)


class RegYelp(Dataset):
    # Universal Dependencies English Web Treebank.
    # Download original at http://universaldependencies.org/
    # License: http://creativecommons.org/licenses/by-sa/4.0/
    urls = []
    dirname = ''
    name = ''

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", verbose=1, shuffle_seed=42, **kwargs):
        examples = []
        n_examples, n_words, n_chars = 0, [], []
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                sen = line.strip().split()
                examples.append(data.Example.fromlist([sen], fields))
                n_examples += 1
                n_words.append(len(sen))
        if verbose:
            print("Dataset has {}  examples. statistics:\n -words: {}+-{}(quantiles(0.5, 0.7, 0.9, 0.95, "
                  "0.99:{},{},{},{},{})".format(n_examples, np.mean(n_words), np.std(n_words),
                                                *np.quantile(n_words, [0.5, 0.7, 0.9, 0.95, 0.99])))
        np.random.seed(42)
        np.random.shuffle(examples)
        super(RegYelp, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root=".data", train="train.tsv",
               validation="valid.tsv",
               test="test.tsv", **kwargs):

        return super(RegYelp, cls).splits(
            path=os.path.join(".data", "yelp_reg"), fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)
