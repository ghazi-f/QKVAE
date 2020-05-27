#!/usr/bin/env bash
python train.py --losses "SSVAE" --test_name "SSVAE/0.03equal" --supervision_proportion 0.03 --batch_size 40 --grad_accu 4
python train.py --losses "SSVAE" --test_name "SSVAE/0.1equal" --supervision_proportion 0.1 ----batch_size 40 grad_accu 4
python train.py --losses "SSVAE" --test_name "SSVAE/0.3equal" --supervision_proportion 0.3 --batch_size 40 --grad_accu 4
python train.py --losses "SSVAE" --test_name "SSVAE/1.0equal" --supervision_proportion 1.0 --batch_size 40 --grad_accu 4