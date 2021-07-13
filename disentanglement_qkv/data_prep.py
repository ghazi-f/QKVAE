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

from datasets import load_dataset
from tokenizers.models import BPE, WordLevel
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import datasets as hdatasets

# ========================================== BATCH ITERATING ENDPOINTS =================================================
VOCAB_LIMIT = 10000


class HuggingYelp2:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>' ,is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, unk_token=None)

        start = time()
        self.divide_bs = 10

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
            (train, val, test), batch_size=int(batch_size/self.divide_bs), device=device, shuffle=False, sort=False)

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


class ParaNMTData:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>' ,is_target=True)

        start = time()

        train, val, test = ParaNMT.splits((('text', text_field), ('para', text_field)))

        # np.random.shuffle(train_examples)
        fields1 = {'text': text_field, 'para': text_field}
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
        self.text_field = text_field
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


class ParaNMTData2:

    def __init__(self, max_len, batch_size, max_epochs, device, unsup_proportion=1., sup_proportion=1., dev_index=1,
                 pretrained=False):

        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs
        self.divide_bs = 1

        np.random.seed(42)
        # Loading Data
        folder = os.path.join(".data", "paranmt")
        train_path, valid_path, test_path = os.path.join(folder, 'train.txt'), os.path.join(folder, 'valid.txt'), \
                                os.path.join(folder, 'test.txt')
        self.dataset = load_dataset('csv', data_files={'train': train_path, 'valid': valid_path, 'test': test_path}
                               , delimiter='\t', column_names=['text', 'para'], keep_in_memory=True)
        # print("Original text Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['text'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Original text length:", np.mean(tr_len))
        # print("Paraphrase Quantiles [0.5, 0.7, 0.9, 0.95, 0.99]")
        # tr_len = [len(ex['para'].split()) for i, ex in enumerate(self.dataset['train']) if i < 50000]
        # print(np.quantile(tr_len, [0.5, 0.7, 0.9, 0.95, 0.99]))
        # print("Mean Paraphrase length:", np.mean(tr_len))
        self.dataset = {'train': self.dataset['train'][:500000],
                        'valid': self.dataset['valid'][:5000],
                        'test': self.dataset['test'][:5000]}
        # load_from_disk()

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

        np.random.shuffle(self.dataset['train']['text']), np.random.shuffle(self.dataset['train']['para'])
        np.random.shuffle(self.dataset['valid']['text']), np.random.shuffle(self.dataset['valid']['para'])
        np.random.shuffle(self.dataset['test']['text']), np.random.shuffle(self.dataset['test']['para'])
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

    def reinit_iterator(self, split):
        if split == 'train':
            self.n_epochs += 1
            print("Finished epoch n°{}".format(self.n_epochs))
            if self.n_epochs < self.max_epochs:
                np.random.shuffle(self.dataset['train']['text']), np.random.shuffle(self.dataset['train']['para'])
                self.train_iter = MyIter(self, self.dataset['train'])
            else:
                print("Reached n_epochs={} and finished training !".format(self.n_epochs))
                self.train_iter = None

        elif split == 'valid':
            np.random.shuffle(self.dataset['valid']['text']), np.random.shuffle(self.dataset['valid']['para'])
            self.val_iter = MyIter(self, self.dataset['valid'], divide_bs=self.divide_bs)
        elif split == 'test':
            np.random.shuffle(self.dataset['test']['text']), np.random.shuffle(self.dataset['test']['para'])
            np.random.shuffle(self.dataset['test'])
            self.test_iter = MyIter(self, self.dataset['test'], divide_bs=self.divide_bs)
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
        folder = os.path.join(".data", "paranmt2")
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
        folder = os.path.join(".data", "paranmt2")
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


class NLIGen(LanguageModelingDataset):

    urls = ['https://raw.githubusercontent.com/schmiflo/crf-generation/master/generated-text/train']
    name = 'nli_gen'
    dirname = 'nli_gen'

    @classmethod
    def splits(cls, text_field, root='.data', train='train.txt',
               validation='valid.txt', test='test.txt',
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


class ParaNMT(Dataset):
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
                # print(line)
                sen, para = line.split('\t')
                sen, para = sen.split(), para.split()
                examples.append(data.Example.fromlist([sen, para], fields))
                n_examples += 1
                n_words.append(len(sen))
        if verbose:
            print("Dataset has {}  examples. statistics:\n -words: {}+-{}(quantiles(0.5, 0.7, 0.9, 0.95, "
                  "0.99:{},{},{},{},{})".format(n_examples, np.mean(n_words), np.std(n_words),
                                                *np.quantile(n_words, [0.5, 0.7, 0.9, 0.95, 0.99])))
        np.random.seed(42)
        np.random.shuffle(examples)
        super(ParaNMT, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root=".data", train="train.txt",
               validation="valid.txt",
               test="test.txt", **kwargs):
        """Loads the Universal Dependencies Version 1 POS Tagged
        data.
        """

        return super(ParaNMT, cls).splits(
            path=os.path.join(".data", "paranmt"), fields=fields, root=root, train=train, validation=validation,
            test=test, **kwargs)
