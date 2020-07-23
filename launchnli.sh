#!/usr/bin/env bash
python nli_train.py --test_name "nlilm/kl1" --kl_th 0.001953125 --device "cuda:2"
python nli_train.py --test_name "nlilm/kl2" --kl_th 0.00390625 --device "cuda:2"
python nli_train.py --test_name "nlilm/kl3" --kl_th 0.005859375 --device "cuda:2"
python nli_train.py --test_name "nlilm/kl4" --kl_th 0.0078125 --device "cuda:2"