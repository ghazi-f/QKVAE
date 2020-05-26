#!/usr/bin/env bash
python train.py --losses "SSVAE" --test_name "SSVAE/UnsupVar0.03" --unsupervision_proportion 0.03 --supervision_proportion 0.03 --grad_accu 16
python train.py --losses "SSVAE" --test_name "SSVAE/UnsupVar0.1" --unsupervision_proportion 0.1 --supervision_proportion 0.03 --grad_accu 16
python train.py --losses "SSVAE" --test_name "SSVAE/UnsupVar0.3" --unsupervision_proportion 0.3 --supervision_proportion 0.03 --grad_accu 16
#python train.py --losses "SSVAE" --test_name "SSVAE/UnsupVar1.0" --unsupervision_proportion 1.0 --supervision_proportion 0.03 --grad_accu 16