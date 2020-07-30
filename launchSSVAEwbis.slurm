#!/usr/bin/env bash
#SBATCH --job-name=python_Ghazi
#SBATCH --gres=gpu:1
#SBATCH --output=./imdbvaeweightedbis.txt
#SBATCH --error=./imdbvaeweightedbis.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.001smallw-2" --generation_weight 0.01 --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.003smallw-2" --generation_weight 0.01 --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01smallw-2" --generation_weight 0.01 --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03smallw-2" --generation_weight 0.01 --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1smallw-2" --generation_weight 0.01 --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3smallw-2" --generation_weight 0.01 --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0smallw-2" --generation_weight 0.01 --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.001smallw-4" --generation_weight 0.0001 --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.003smallw-4" --generation_weight 0.0001 --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01smallw-4" --generation_weight 0.0001 --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03smallw-4" --generation_weight 0.0001 --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1smallw-4" --generation_weight 0.0001 --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3smallw-4" --generation_weight 0.0001 --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0smallw-4" --generation_weight 0.0001 --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.001smallw-6" --generation_weight 0.000001 --supervision_proportion 0.001 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.003smallw-6" --generation_weight 0.000001 --supervision_proportion 0.003 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.01smallw-6" --generation_weight 0.000001 --supervision_proportion 0.01 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03smallw-6" --generation_weight 0.000001 --supervision_proportion 0.03 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1smallw-6" --generation_weight 0.000001 --supervision_proportion 0.1 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3smallw-6" --generation_weight 0.000001 --supervision_proportion 0.3 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
# srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0smallw-6" --generation_weight 0.000001 --supervision_proportion 1.0 --embedding_dim 200 --z_size 100 --text_rep_h 200 --encoder_h 200 --decoder_h 200 --pos_embedding_dim 100 --pos_h 100
