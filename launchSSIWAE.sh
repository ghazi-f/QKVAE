#!/usr/bin/env bash
python sent_train.py --losses "SSIWAE" --test_name "IMDB/SSIWAE/0.03" --supervision_proportion 0.03 --grad_accu 8 --batch_size 8 --testing_iw_samples 10 --device "cuda:2"
python sent_train.py --losses "SSIWAE" --test_name "IMDB/SSIWAE/0.1" --supervision_proportion 0.1 --grad_accu 8 --batch_size 8 --testing_iw_samples 10 --device "cuda:2"
python sent_train.py --losses "SSIWAE" --test_name "IMDB/SSIWAE/0.3" --supervision_proportion 0.3 --grad_accu 8 --batch_size 8 --testing_iw_samples 10 --device "cuda:2"
python sent_train.py --losses "SSIWAE" --test_name "IMDB/SSIWAE/1.0" --supervision_proportion 1.0 --grad_accu 8 --batch_size 8 --testing_iw_samples 10 --device "cuda:2"