# This file will implement the main training loop for a model
from time import time

from torch import device
import torch
import numpy as np

from data_prep import Wiki2Data as Data
from pos_tagging.models import SSPoSTag as Model
from pos_tagging.h_params import DefaultSSVariationalHParams as HParams
from pos_tagging.vertices import *

MAX_LEN = 20
BATCH_SIZE = 64
N_EPOCHS = 200
TEST_FREQ = 10
COMPLETE_TEST_FREQ = TEST_FREQ * 5
DEVICE = device('cuda:0')

data = Data(MAX_LEN, BATCH_SIZE, N_EPOCHS, DEVICE)
h_params = HParams(len(data.vocab.itos), 0, MAX_LEN, BATCH_SIZE, N_EPOCHS,
                   device=DEVICE, pos_ignore_index=None,
                   vocab_ignore_index=data.vocab.stoi['<pad>'], decoder_h=1000, decoder_l=6, encoder_h=1000, encoder_l=6,
                   test_name='Wikigen/nlimin14', grad_accumulation_steps=1, optimizer_kwargs={'lr': 1e-3},
                   is_weighted=[], graph_generator=get_graph_vsl, z_size=200, embedding_dim=600, anneal_kl=[1000, 4500],
                   grad_clip=10., kl_th=1/1000,
                   highway=True)
val_iterator = iter(data.val_iter)
print("Words: ", len(data.vocab.itos), "Target tags: ", 0, " On device: ", DEVICE.type)
model = Model(data.vocab, data.tags, h_params, wvs=data.wvs)
if DEVICE.type == 'cuda':
    model.cuda(DEVICE)

current_time = time()
replace = False
#print(model)

print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
min_loss = 1e10
last_min_count = 0
waiting_time = 1000

while data.train_iter is not None:
    for training_batch in data.train_iter:
        if replace:
            batch = training_batch
            replace = False
            interest = torch.unique(training_batch.text).view(-1)
        else:
            pass
            # training_batch = batch
        loss = model.opt_step({'x': training_batch.text})
        if min_loss > loss:
            min_loss = loss
            last_min_count = 0
        else:
            last_min_count += 1

        if last_min_count > waiting_time:
            last_min_count = 0
            model.reduce_lr(10.)
            waiting_time *= 10
            print('Learning rate reduced to ', [gr['lr'] for gr in model.optimizer.param_groups])
        print("step:{}, loss:{}, seconds/step:{}".format(model.step, loss, time()-current_time))
        if model.step % TEST_FREQ == TEST_FREQ-1:
            try:
                test_batch = next(val_iterator)
            except StopIteration:
                print("Reinitialized test data iterator")
                val_iterator = iter(data.val_iter)
                test_batch = next(val_iterator)
            model({'x': test_batch.text})
            model.dump_test_viz(complete=model.step % COMPLETE_TEST_FREQ == COMPLETE_TEST_FREQ-1)

        current_time = time()
        if model.step % (COMPLETE_TEST_FREQ*10) == (COMPLETE_TEST_FREQ *10 - 1):
            '''print(torch.max(model.gen_bn.variables_hat[model.generated_v][:, :, interest], dim=-1)[0]/
                  torch.max(model.gen_bn.variables_hat[model.generated_v], dim=-1)[0])
            print([[data.vocab.itos[t] for t in sent] for sent in torch.argmax(model.gen_bn.variables_hat[model.generated_v], dim=-1)])'''
            #print([' '.join([data.vocab.itos[t] for t in text_i]) for text_i in training_batch.text])
            print('Saving The model ..')
            if not torch.isnan(loss):
                model.save()

    data.reinit_iterator('valid')
    model.get_perplexity(data.val_iter)
    data.reinit_iterator('valid')
    data.reinit_iterator('train')

