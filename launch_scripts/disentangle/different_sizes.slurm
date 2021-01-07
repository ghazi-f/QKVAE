#!/usr/bin/env bash
#SBATCH --job-name=DisLow
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./disentangle_sizes.txt
#SBATCH --error=./disentangle_sizes.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Unweighted experiments
srun python disentangle_train.py --test_name "Disentanglement/low_variables1" --n_latents 8 8 8 --n_epochs 40
srun python disentangle_train.py --test_name "Disentanglement/low_variables2" --n_latents 8 8 8 --n_epochs 40
srun python disentangle_train.py --test_name "Disentanglement/low_variables3" --n_latents 8 8 8 --n_epochs 40
srun python disentangle_train.py --test_name "Disentanglement/low_variables4" --n_latents 8 8 8 --n_epochs 40
srun python disentangle_train.py --test_name "Disentanglement/low_variables5" --n_latents 8 8 8 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/high_variables1" --n_latents 32 32 32 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/high_variables2" --n_latents 32 32 32 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/high_variables3" --n_latents 32 32 32 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/high_variables4" --n_latents 32 32 32 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/high_variables5" --n_latents 32 32 32 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/deep_variables1" --n_latents 16 16 16 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/deep_variables2" --n_latents 16 16 16 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/deep_variables3" --n_latents 16 16 16 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/deep_variables4" --n_latents 16 16 16 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/deep_variables5" --n_latents 16 16 16 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/shallow_variables1" --n_latents 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/shallow_variables2" --n_latents 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/shallow_variables3" --n_latents 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/shallow_variables4" --n_latents 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/shallow_variables5" --n_latents 16 16 --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/minimal_enc1" --minimal_enc --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/minimal_enc2" --minimal_enc --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/minimal_enc3" --minimal_enc --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/minimal_enc4" --minimal_enc --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/minimal_enc5" --minimal_enc --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/simple_prior1" --graph NormalSimplePrior --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/simple_prior2" --graph NormalSimplePrior --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/simple_prior3" --graph NormalSimplePrior --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/simple_prior4" --graph NormalSimplePrior --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/simple_prior5" --graph NormalSimplePrior --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/con_gen1" --graph NormalConGen --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/con_gen2" --graph NormalConGen --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/con_gen3" --graph NormalConGen --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/con_gen4" --graph NormalConGen --n_epochs 40
#srun python disentangle_train.py --test_name "Disentanglement/con_gen5" --graph NormalConGen --n_epochs 40

