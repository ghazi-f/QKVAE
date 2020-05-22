#!/usr/bin/env bash
python train.py --losses "SSVAE" --test_name "SSVAE/0.03" --supervision_proportion 0.03 --grad_accu 16
python train.py --losses "SSVAE" --test_name "SSVAE/0.1" --supervision_proportion 0.1 --grad_accu 16
python train.py --losses "SSVAE" --test_name "SSVAE/0.3" --supervision_proportion 0.3 --grad_accu 16
python train.py --losses "SSVAE" --test_name "SSVAE/1.0" --supervision_proportion 1.0 --grad_accu 16