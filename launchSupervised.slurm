#!/usr/bin/env bash
#SBATCH --job-name=python_Ghazi
#SBATCH --gres=gpu:1
#SBATCH --output=./imdbsupsmall.txt
#SBATCH --error=./imdbsupsmall.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.03small" --supervision_proportion 0.03
srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.1small" --supervision_proportion 0.1
srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/0.3small" --supervision_proportion 0.3
srun python sent_train.py --losses "S" --test_name "IMDB/Supervised/1.0small" --supervision_proportion 1.0