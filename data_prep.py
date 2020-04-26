# This file is destined to wrap all the data pipelining utilities (reading, tokenizing, padding, batchifying .. )

import torchtext
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import FastText


# ========================================== BATCH ITERATING ENDPOINTS =================================================
class IMDBData:
    def __init__(self, max_len, batch_size, device):
        text_field = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=max_len)
        label_field = data.Field(sequential=False)

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
        text_field = data.Field(lower=True, batch_first=True, fix_length=max_len)
        label_field = data.Field(sequential=True, fix_length=max_len, batch_first=True)

        # make splits for data
        train, val, test = datasets.UDPOS.splits((('text', text_field), ('label', label_field)))

        # build the vocabulary
        text_field.build_vocab(train)  # , vectors="fasttext.simple.300d")
        label_field.build_vocab(train)

        # make iterator for splits
        self.train_iter, self.val_iter,  self.test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device, shuffle=True, sort=False)
        self.val_iter.shuffle = False
        self.test_iter.shuffle = False

        self.vocab = text_field.vocab
        self.tags = label_field.vocab
        self.text_field = text_field
        self.label_field = label_field
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = 0
        self.max_epochs = max_epochs

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
