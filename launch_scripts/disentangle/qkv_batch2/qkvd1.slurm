#!/usr/bin/env bash
#SBATCH --job-name=QKVXL1
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVXL1.txt
#SBATCH --error=./QKVXL1.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1

srun python3 qkv_train.py --test_name "Disentanglement2/PQBmTXL1_beta0.6.0.6.1.4" --kl_beta 0.6 --n_latents 4 --data paranmt\
 --n_epochs 40 --graph QKV --max_len 40 --kl_beta_zs 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.05 \
 --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --layer_wise_qkv --grad_clip 100.0

