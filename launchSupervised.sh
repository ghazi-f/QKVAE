#!/usr/bin/env bash
python train.py --losses "S" --test_name "Supervised/0.03" --supervision_proportion 0.03 --device "cuda:1"
python train.py --losses "S" --test_name "Supervised/0.1" --supervision_proportion 0.1 --device "cuda:1"
python train.py --losses "S" --test_name "Supervised/0.3" --supervision_proportion 0.3 --device "cuda:1"
python train.py --losses "S" --test_name "Supervised/1.0" --supervision_proportion 1.0 --device "cuda:1"