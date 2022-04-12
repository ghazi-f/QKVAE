#!/usr/bin/env bash
#SBATCH --job-name=PADVAEmini64_3
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./PADVAEmini64_3.txt
#SBATCH --error=./PADVAEmini64_3.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1

srun python qkv_train.py --test_name "Disentanglement2/PADVAE_min3_beta0.6.0.3.1.4" --kl_beta 0.6 --n_latents 4 --data paranmt\
 --n_epochs 40 --graph IndepInfer --max_len 40 --kl_beta_zs 0.3 --use_bart --batch_size 64 --z_size 1536 --kl_th 0.01 \
 --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --grad_clip 100.0 --bart_l 4

