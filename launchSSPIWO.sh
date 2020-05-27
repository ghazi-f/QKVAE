#!/usr/bin/env bash
python train.py --losses "SSPIWO" --test_name "SSPIWO/0.03equal" --supervision_proportion 0.03 --grad_accu 10 --batch_size 16
python train.py --losses "SSPIWO" --test_name "SSPIWO/0.1equal" --supervision_proportion 0.1 --grad_accu 10 --batch_size 16
python train.py --losses "SSPIWO" --test_name "SSPIWO/0.3equal" --supervision_proportion 0.3 --grad_accu 10 --batch_size 16
python train.py --losses "SSPIWO" --test_name "SSPIWO/1.0equal" --supervision_proportion 1.0 --grad_accu 10 --batch_size 16