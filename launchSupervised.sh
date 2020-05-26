#!/usr/bin/env bash
python train.py --losses "S" --test_name "Supervised/0.03Corr" --supervision_proportion 0.03
python train.py --losses "S" --test_name "Supervised/0.1Corr" --supervision_proportion 0.1
python train.py --losses "S" --test_name "Supervised/0.3Corr" --supervision_proportion 0.3
python train.py --losses "S" --test_name "Supervised/1.0Corr" --supervision_proportion 1.0