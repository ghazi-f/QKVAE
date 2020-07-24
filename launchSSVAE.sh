#!/usr/bin/env bash
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.001" --supervision_proportion 0.001
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.003" --supervision_proportion 0.003
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01" --supervision_proportion 0.01
#python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03" --supervision_proportion 0.03
#python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1" --supervision_proportion 0.1
#python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3" --supervision_proportion 0.3
#python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0" --supervision_proportion 1.0