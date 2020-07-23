#!/usr/bin/env bash
#SBATCH --job-name=python_Ghazi
#SBATCH --gres=gpu:1
#SBATCH --output=./imdbvaew.txt
#SBATCH --error=./imdbvaew.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.03small" --supervision_proportion 0.03
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.1small" --supervision_proportion 0.1
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/0.3small" --supervision_proportion 0.3
srun python sent_train.py --losses "SSVAE" --test_name "IMDB/SSVAE/1.0small" --supervision_proportion 1.0