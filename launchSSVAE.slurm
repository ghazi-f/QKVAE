#!/usr/bin/env bash
#SBATCH --job-name=python_Ghazi
#SBATCH --gres=gpu:1
#SBATCH --output=./imdbvaewsizes3.txt
#SBATCH --error=./imdbvaewsizes3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.001smallest" --supervision_proportion 0.001 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.003smallest" --supervision_proportion 0.003 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01smallest" --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03smallest" --supervision_proportion 0.03 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1smallest" --supervision_proportion 0.1 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3smallest" --supervision_proportion 0.3 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
#srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0smallest" --supervision_proportion 1.0 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01size1uns2" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 20 --z_size 20 --text_rep_h 20 --encoder_h 20 --decoder_h 20 --pos_embedding_dim 10 --pos_h 10
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01size2uns2" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 40 --z_size 40 --text_rep_h 40 --encoder_h 40 --decoder_h 40 --pos_embedding_dim 20 --pos_h 20
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01size3uns2" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 80 --z_size 80 --text_rep_h 80 --encoder_h 80 --decoder_h 80 --pos_embedding_dim 40 --pos_h 40
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01size4uns2" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 160 --z_size 160 --text_rep_h 160 --encoder_h 160 --decoder_h 160 --pos_embedding_dim 80 --pos_h 80
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01size5uns2" --unsupervision_proportion 0.01 --supervision_proportion 0.01 --embedding_dim 320 --z_size 320 --text_rep_h 320 --encoder_h 320 --decoder_h 320 --pos_embedding_dim 160 --pos_h 160
