#!/usr/bin/env bash
#SBATCH --job-name=python_Ghazi
#SBATCH --gres=gpu:1
#SBATCH --output=./imdbsupsmall.txt
#SBATCH --error=./imdbsupsmall.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.001small" --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200
srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.003small" --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200
srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.01small" --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200
#srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.03small" --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200
#srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.1small" --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200
#srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.3small" --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200
#srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/1.0small" --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200