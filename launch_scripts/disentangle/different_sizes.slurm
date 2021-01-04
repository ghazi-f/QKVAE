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
#srun python disentangle_train.py --test_name "Disentanglement/low_variables" --n_latents 8 8 8
#srun python disentangle_train.py --test_name "Disentanglement/high_variables" --n_latents 32 32 32
#srun python disentangle_train.py --test_name "Disentanglement/deep_variables" --n_latents 16 16 16 16 16
#srun python disentangle_train.py --test_name "Disentanglement/shallow_variables" --n_latents 16 16
srun python disentangle_train.py --test_name "Disentanglement/reproduce11" --n_latents 16 16 16 --n_epochs 40
srun python disentangle_train.py --test_name "Disentanglement/reproduce11wd" --n_latents 16 16 16 --n_epochs 40 --word_dropout 0.3
srun python disentangle_train.py --test_name "Disentanglement/reproduce12" --n_latents 16 16 16 --n_epochs 40
srun python disentangle_train.py --test_name "Disentanglement/reproduce12wd" --n_latents 16 16 16 --n_epochs 40 --word_dropout 0.3
srun python disentangle_train.py --test_name "Disentanglement/reproduce13" --n_latents 16 16 16 --n_epochs 40
srun python disentangle_train.py --test_name "Disentanglement/reproduce13wd" --n_latents 16 16 16 --n_epochs 40 --word_dropout 0.3
#srun python disentangle_train.py --test_name "Disentanglement/minimal_enc" --minimal_enc
#srun python disentangle_train.py --test_name "Disentanglement/simple_prior" --graph NormalSimplePrior
#srun python disentangle_train.py --test_name "Disentanglement/con_gen" --graph NormalConGen

