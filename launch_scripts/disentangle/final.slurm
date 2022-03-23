#!/usr/bin/env bash
#SBATCH --job-name=NQKm5
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./NQKm5.txt
#SBATCH --error=./NQKm5.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH -A jrp@gpu


#srun python final_disent.py --test_name "Disentanglement2/BCNX_beta0.6.0.6.1.4" --kl_beta 0.6 --n_latents 4 --data bc\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --use_bart --batch_size 32 --z_size 768 --kl_th 0.05 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --layer_wise_qkv --grad_clip 100.0 --sem_coeff 100.0

#srun python final_disent.py --test_name "Disentanglement2/BCNmini64_beta0.6.0.6.1.4" --kl_beta 0.6 --n_latents 4 --data bc\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.05 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --layer_wise_qkv --grad_clip 100.0 --sem_coeff 100.0 --bart_l 3

srun python final_disent.py --test_name "Disentanglement2/OWTNm5_beta1.0.1.0.1.4" --kl_beta 1.0 --n_latents 4 --data owt\
 --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 1.0 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.05 \
 --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 7000 --zg_anneal_kl1 20000 --layer_wise_qkv --grad_clip 100.0 \
 --sem_coeff 100.0 --bart_l 3 --aux_l 3