#!/usr/bin/env bash
#SBATCH --job-name=python_Ghazi
#SBATCH --gres=gpu:1
#SBATCH --output=./imdbvaewsmallest.txt
#SBATCH --error=./imdbvaewsmallest.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.001smallest" --supervision_proportion 0.001 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.003smallest" --supervision_proportion 0.003 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01smallest" --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03smallest" --supervision_proportion 0.03 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1smallest" --supervision_proportion 0.1 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3smallest" --supervision_proportion 0.3 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0smallest" --supervision_proportion 1.0 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10