# This file will implement the main training loop for a model
from time import time

from torch import device

from data_prep import UDPoSDaTA as Data
from models import SSVAE
from h_params import DefaultSSVariationalHParams as HParams

MAX_LEN = 40
BATCH_SIZE = 128
N_EPOCHS = 100
TEST_FREQ = 5
COMPLETE_TEST_FREQ = TEST_FREQ * 10
DEVICE = device('cuda:0')

data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE)
h_params = HParams(len(data.vocab.itos), MAX_LEN, BATCH_SIZE, N_EPOCHS, len(data.tags.itos),
                   device=DEVICE, token_ignore_index=data.tags.stoi['<pad>'],
                   target_ignore_index=data.vocab.stoi['<pad>'], decoder_h=512, decoder_l=3, encoder_h=512, encoder_l=3,
                   test_name='SSVbigger')
test_iterator = data.test_iter.data()
print("words: ", len(data.vocab.itos), "Target tags: ", len(data.tags.itos), " On device: ", DEVICE.type)
model = SSVAE(data.vocab, data.tags, h_params)
if DEVICE.type == 'cuda':
    model.cuda(DEVICE)

current_time = time()
while data.train_iter is not None:
    for training_batch in data.train_iter:
        loss = model.opt_step([training_batch.text, training_batch.label])
        print("step:{}, loss:{}, seconds/step:{}".format(model.step, loss, time()-current_time))
        if model.step % TEST_FREQ == TEST_FREQ-1:
            try:
                test_batch = next(iter(data.test_iter))
            except StopIteration:
                print("Reinitialized test data iterator")
                data.reinit_iterator('test')
                test_batch = next(iter(data.test_iter))
            model([test_batch.text, test_batch.label], is_training=False)
            model.dump_test_viz(complete=model.step % COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)
        current_time = time()
        if model.step % COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ - 1:
            print('Saving The model ..')
            model.save()
    data.reinit_iterator('valid')
    model.get_perplexity(data.val_iter)
    data.reinit_iterator('valid')
    data.reinit_iterator('train')
