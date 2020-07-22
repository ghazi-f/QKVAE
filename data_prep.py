# This file is destined to wrap all the data pipelining utilities (reading, tokenizing, padding, batchifying .. )
import io

import torchtext.data as data
from torchtext.data import Dataset, Example
import torchtext.datasets as datasets
from torchtext.vocab import FastText
import numpy as np
from time import time

from nlp import load_dataset


# ========================================== BATCH ITERATING ENDPOINTS =================================================

class HuggingIMDB2:
    def __init__(self, max_len, batch_size, max_epochs, device):
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len, pad_token='<pad>',
                                init_token='<go>'
                                ,
                                is_target=True)  # init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len - 1, batch_first=True, unk_token=None)

        start = time()
        train_data, test_data, unsup_data = load_dataset('imdb')['train'], load_dataset('imdb')['test'],\
                                            load_dataset('imdb')['unsupervised']

        def expand_labels(data):
            # data['label'] = ' '.join([str(data['label'])]*(max_len-1))
            data['label'] = [str(data['label'])]*(max_len-1)
            return data
        train_data, test_data = train_data.map(expand_labels), test_data.map(expand_labels)
        fields1 = {'text': text_field, 'label': label_field}
        fields2 = {'text': ('text', text_field), 'label': ('label', label_field)}
        fields3 = {'text': text_field}
        fields4 = {'text': ('text', text_field)}
        train = Dataset([Example.fromdict(ex, fields2) for ex in train_data], fields1)
        test = Dataset([Example.fromdict(ex, fields2) for ex in test_data], fields1)
        unsup_train = Dataset([Example.fromdict(ex, fields4) for ex in unsup_data] +
                              [Example.fromdict(ex, fields4) for ex in train_data]
                              , fields3)

        val, unsup_test, unsup_val = test, test, test

        print('data loading took', time() - start)

        # build the vocabulary
        text_field.build_vocab(unsup_train, max_size=10000)  # , vectors="fasttext.simple.300d")
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
    def __init__(self, max_len, batch_size, max_epochs, device):
        text_field = data.Field(lower=True, batch_first=True,  fix_length=max_len, pad_token='<pad>', init_token='<eos>'
                                , is_target=True)#init_token='<go>', eos_token='<eos>', unk_token='<unk>', pad_token='<unk>')
        label_field = data.Field(fix_length=max_len-1, batch_first=True)

        # make splits for data
        #unsup_train, unsup_val, unsup_test = MyPennTreebank.splits(text_field)
        #unsup_train, unsup_val, unsup_test = datasets.PennTreebank.splits(text_field)
        #unsup_train, unsup_val, unsup_test = datasets.WikiText2.splits(text_field)
        #unsup_train, unsup_val, unsup_test = datasets.UDPOS.splits((('text', text_field), ('label', label_field)))
        unsup_train, unsup_val, unsup_test = YahooLM.splits(text_field)
        train, val, test = datasets.UDPOS.splits((('text', text_field), ('label', label_field)))

        # build the vocabulary
        text_field.build_vocab(unsup_train, max_size=30000)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)
        # self.train_iter, _,  _ = data.BPTTIterator.splits((unsup_train, unsup_val, unsup_test),
        #                                                                     batch_size=batch_size, bptt_len=max_len,
        #                                                                     device=device, repeat=False, shuffle=False,
        #                                                                     sort=False)
        # _, self.unsup_val_iter,  _ = data.BPTTIterator.splits((unsup_train, unsup_val, unsup_test),
        #                                                                     batch_size=int(batch_size/10), bptt_len=max_len,
        #                                                                     device=device, repeat=False, shuffle=False,
        #                                                                     sort=False)

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
    def __init__(self, max_len, batch_size, max_epochs, device):
        text_field = data.Field(lower=True, batch_first=True,  fix_length=max_len, init_token='<eos>', eos_token='<eos>',
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
                  np.quantile(seq_lens, [0.25, 0.5, 0.7, 0.9, 0.95, 0.99]))

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
