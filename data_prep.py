# This file is destined to wrap all the data pipelining utilities (reading, tokenizing, padding, batchifying .. )
import io

import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import FastText


# ========================================== BATCH ITERATING ENDPOINTS =================================================
class IMDBData:
    def __init__(self, max_len, batch_size, device):
        text_field = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=max_len)
        label_field = data.Field(sequential=False, fix_length=max_len-1)

        # make splits for data
        train, test = datasets.IMDB.splits(text_field, label_field)

        # build the vocabulary
        text_field.build_vocab(train)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits
        self.train_iter, self.test_iter = data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
        self.vocab = text_field.vocab


class UDPoSDaTA:
    def __init__(self, max_len, batch_size, max_epochs, device):
        text_field = data.Field(lower=True, batch_first=True,  # fix_length=max_len,
                                init_token='<go>', eos_token='<eos>')
        label_field = data.Field(sequential=True,  # fix_length=max_len-2,
                                 batch_first=True)

        # make splits for data
        train, val, test = datasets.UDPOS.splits((('text', text_field), ('label', label_field)))

        # build the vocabulary
        text_field.build_vocab(train)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits
        self.train_iter, _,  _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=True)
        self.sup_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=False, sort=False)
        _, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=int(batch_size/4), device=device, shuffle=False, sort=False)

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
        #ftxt = FastText()
        self.wvs = None#ftxt.get_vecs_by_tokens(self.vocab.itos)

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
            for line in f:
                processed_line = text_field.preprocess(line)
                seq_lens.append(len(processed_line))
                if len(processed_line) > 1:
                    examples.append(data.Example.fromlist([processed_line], fields))
                #break
        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)


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
