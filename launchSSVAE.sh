#!/usr/bin/env bash
#python train.py --losses "SSVAE" --test_name "SSVAE/0.03equal" --supervision_proportion 0.03 --batch_size 80 --grad_accu 2
#python train.py --losses "SSVAE" --test_name "SSVAE/0.1equal" --supervision_proportion 0.1 ----batch_size 80 grad_accu 2
#python train.py --losses "SSVAE" --test_name "SSVAE/0.3equal" --supervision_proportion 0.3 --batch_size 80 --grad_accu 2
#python train.py --losses "SSVAE" --test_name "SSVAE/1.0equal" --supervision_proportion 1.0 --batch_size 80 --grad_accu 2
python train.py --losses "SSVAE" --test_name "SSVAE/0.03x1e0" --supervision_proportion 0.03 --generation_weight 1.0 --grad_accu 2 --batch_size 80
python train.py --losses "SSVAE" --test_name "SSVAE/1.x1e0" --supervision_proportion 1.0 --generation_weight 1.0 --grad_accu 2 --batch_size 80
python train.py --losses "SSVAE" --test_name "SSVAE/0.03x1e-2" --supervision_proportion 0.03 --generation_weight 0.01 --grad_accu 2 --batch_size 80
python train.py --losses "SSVAE" --test_name "SSVAE/1.x1e-2" --supervision_proportion 1.0 --generation_weight 0.01 --grad_accu 2 --batch_size 80
python train.py --losses "SSVAE" --test_name "SSVAE/0.03x1e-4" --supervision_proportion 0.03 --generation_weight 0.0001 --grad_accu 2 --batch_size 80
python train.py --losses "SSVAE" --test_name "SSVAE/1.x1e-4" --supervision_proportion 1.0 --generation_weight 0.0001 --grad_accu 2 --batch_size 80
python train.py --losses "SSVAE" --test_name "SSVAE/0.03x1e-6" --supervision_proportion 0.03 --generation_weight 0.000001 --grad_accu 2 --batch_size 80
python train.py --losses "SSVAE" --test_name "SSVAE/1.x1e-6" --supervision_proportion 1.0 --generation_weight 0.000001 --grad_accu 2 --batch_size 80
