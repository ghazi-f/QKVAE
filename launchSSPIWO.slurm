#!/usr/bin/env bash
#SBATCH --job-name=python_Ghazi
#SBATCH --gres=gpu:1
#SBATCH --output=./imdbpiwosmall.txt
#SBATCH --error=./imdbpiwosmall.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
srun python sent_train.py --losses "SSPIWO" --test_name "IMDB/SSPIWO/0.03small" --supervision_proportion 0.03
srun python sent_train.py --losses "SSPIWO" --test_name "IMDB/SSPIWO/0.1small" --supervision_proportion 0.1
srun python sent_train.py --losses "SSPIWO" --test_name "IMDB/SSPIWO/0.3small" --supervision_proportion 0.3
srun python sent_train.py --losses "SSPIWO" --test_name "IMDB/SSPIWO/1.0small" --supervision_proportion 1.0