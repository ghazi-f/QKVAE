#!/usr/bin/env bash
#SBATCH --job-name=Disentangle_sizes
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./disentangle_sizes.txt
#SBATCH --error=./disentangle_sizes.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Unweighted experiments
srun python disentangle_train.py --test_name "Disentanglement/low_variables" --n_latents 8 8 8
srun python disentangle_train.py --test_name "Disentanglement/high_variables" --n_latents 32 32 32
srun python disentangle_train.py --test_name "Disentanglement/deep_variables" --n_latents 16 16 16 16 16
#srun python disentangle_train.py --test_name "Disentanglement/shallow_variables" --n_latents 16 16
#srun python disentangle_train.py --test_name "Disentanglement/regular_variables" --n_latents 16 16 16
#srun python disentangle_train.py --test_name "Disentanglement/minimal_enc" --minimal_enc
#srun python disentangle_train.py --test_name "Disentanglement/simple_prior" --graph NormalSimplePrior
#srun python disentangle_train.py --test_name "Disentanglement/con_gen" --graph NormalConGen

