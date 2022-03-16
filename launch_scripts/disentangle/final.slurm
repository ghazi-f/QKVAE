#!/usr/bin/env bash
#SBATCH --job-name=NQK1
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./NQK1.txt
#SBATCH --error=./NQK1.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1


srun python final_disent.py --test_name "Disentanglement2/BCNX_beta0.6.0.6.1.4" --kl_beta 0.6 --n_latents 4 --data bc\
 --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --use_bart --batch_size 32 --z_size 768 --kl_th 0.05 \
 --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --layer_wise_qkv --grad_clip 100.0 --sem_coeff 100.0