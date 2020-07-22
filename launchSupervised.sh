#!/usr/bin/env bash
python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.03" --supervision_proportion 0.03
python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.1" --supervision_proportion 0.1
python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.3" --supervision_proportion 0.3
python sent_train.py --losses "S" --test_name "IMDB/Supervised/1.0" --supervision_proportion 1.0