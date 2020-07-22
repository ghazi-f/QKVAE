#!/usr/bin/env bash
python train.py --losses "S" --test_name "IMDB/Supervised/0.03" --supervision_proportion 0.03
python train.py --losses "S" --test_name "IMDB/Supervised/0.1" --supervision_proportion 0.1
python train.py --losses "S" --test_name "IMDB/Supervised/0.3" --supervision_proportion 0.3
python train.py --losses "S" --test_name "IMDB/Supervised/1.0" --supervision_proportion 1.0