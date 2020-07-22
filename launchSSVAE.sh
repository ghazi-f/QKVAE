#!/usr/bin/env bash
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03" --supervision_proportion 0.03
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1" --supervision_proportion 0.1
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3" --supervision_proportion 0.3
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0" --supervision_proportion 1.0
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03w" --supervision_proportion 0.03 --generation_weight 0.001
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1w" --supervision_proportion 0.1 --generation_weight 0.001
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3w" --supervision_proportion 0.3 --generation_weight 0.001
python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0w" --supervision_proportion 1.0 --generation_weight 0.001