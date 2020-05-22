#!/usr/bin/env bash
python train.py --losses "SSPIWO" --test_name "SSPIWO/0.03" --supervision_proportion 0.03
python train.py --losses "SSPIWO" --test_name "SSPIWO/0.1" --supervision_proportion 0.1
python train.py --losses "SSPIWO" --test_name "SSPIWO/0.3" --supervision_proportion 0.3
python train.py --losses "SSPIWO" --test_name "SSPIWO/1.0" --supervision_proportion 1.0