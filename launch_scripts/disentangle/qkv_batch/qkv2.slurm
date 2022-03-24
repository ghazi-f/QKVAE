#!/usr/bin/env bash
#SBATCH --job-name=QKVXmini64_2
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVXmini64_2.txt
#SBATCH --error=./QKVXmini64_2.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1

srun python qkv_train.py --test_name "Disentanglement2/PQBmTXLmini64_2_beta0.8.0.6.1.4" --kl_beta 0.8 --n_latents 4 --data paranmt\
 --n_epochs 40 --graph QKV --max_len 40 --kl_beta_zs 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.05 \
 --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --layer_wise_qkv --grad_clip 100.0 --bart_l 3


