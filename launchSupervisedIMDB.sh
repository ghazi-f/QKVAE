#!/usr/bin/env bash

# Unweighted experiments
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.001small1" --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.001small2" --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.001small3" --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.001small4" --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.001small5" --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.003small1" --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.003small2" --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.003small3" --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.003small4" --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.003small5" --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01small1" --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01small2" --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01small3" --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01small4" --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01small5" --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.03small1" --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.03small2" --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.03small3" --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.03small4" --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.03small5" --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.1small1" --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.1small2" --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.1small3" --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.1small4" --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.1small5" --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.3small1" --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.3small2" --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.3small3" --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.3small4" --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.3small5" --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/1.0small1" --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/1.0small2" --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/1.0small3" --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/1.0small4" --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/1.0small5" --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
# Size experiments
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns41" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns42" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns43" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns44" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns45" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns41" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns42" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns43" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns44" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns45" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns41" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns42" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns43" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns44" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns45" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns41" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns42" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns43" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns44" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns45" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns41" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns42" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns43" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns44" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns45" --unsupervision_proportion 1.0 --supervision_proportion 0.003 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns21" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns22" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns23" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns24" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns25" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns21" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns22" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns23" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns24" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns25" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns21" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns22" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns23" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns24" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns25" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns21" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns22" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns23" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns24" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns25" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns21" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns22" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns23" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns24" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns25" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns31" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns32" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns33" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns34" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size1uns35" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns31" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns32" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns33" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns34" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size2uns35" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns31" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns32" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns33" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns34" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size3uns35" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns31" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns32" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns33" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns34" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size4uns35" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns31" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 1 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns32" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 2 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns33" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 3 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns34" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 4 --dataset imdb --result_csv imdb2.csv
python sent_train.py --losses "S" --test_name "IMDB2/Supervised/0.01size5uns35" --unsupervision_proportion 0.01 --supervision_proportion 1.0 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160 --device "cuda:2" --dev_index 5 --dataset imdb --result_csv imdb2.csv
