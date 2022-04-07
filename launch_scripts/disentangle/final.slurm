#!/usr/bin/env bash
#SBATCH --job-name=OWTminii5
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./OWTminii5.txt
#SBATCH --error=./OWTminii5.err
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

#srun python final_disent.py --test_name "Disentanglement2/OWTNm7_beta1.0.1.0.1.4" --kl_beta 1.0 --n_latents 4 --data owt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.8 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --sem_coeff 100.0 --bart_l 3 --aux_l 3

#srun python final_disent.py --test_name "Disentanglement2/OWTNm8_beta1.0.1.0.1.4" --kl_beta 1.0 --n_latents 4 --data owt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.8 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.05 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --sem_coeff 100.0 --bart_l 3 --aux_l 3

# srun python final_disent.py --test_name "Disentanglement2/OWTNm9_beta1.0.1.0.1.4" --kl_beta 1.0 --n_latents 4 --data owt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --sem_coeff 100.0 --bart_l 3 --aux_l 3

#srun python final_disent.py --test_name "Disentanglement2/OWTNm10_beta1.0.1.0.1.4" --kl_beta 0.6 --n_latents 4 --data owt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --sem_coeff 100.0 --bart_l 3 --aux_l 3

#srun python final_disent.py --test_name "Disentanglement2/PNm7_beta1.0.1.0.1.4" --kl_beta 0.6 --n_latents 4 --data paranmt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --bart_l 3 --aux_l 3


#srun python final_disent.py --test_name "Disentanglement2/PNm8_beta1.0.1.0.1.4" --kl_beta 1.0 --n_latents 4 --data paranmt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --bart_l 3 --aux_l 3


#srun python final_disent.py --test_name "Disentanglement2/PNmNoID_beta0.6.0.3.01.4" --kl_beta 0.6 --n_latents 4 --data paranmt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.3 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --bart_l 3 --aux_l 3 --no-z_ids

#srun python final_disent.py --test_name "Disentanglement2/PNminii_beta0.6.0.3.01.4" --kl_beta 0.6 --n_latents 4 --data paranmt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.1 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --bart_l 2 --aux_l 3

# srun python final_disent.py --test_name "Disentanglement2/OWminii_beta0.6.0.6.01.4" --kl_beta 0.6 --n_latents 4 --data owt\
# --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.6 --kl_beta_zg 0.6 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
# --zs_anneal_kl0 7000 --zs_anneal_kl1 20000 --zg_anneal_kl0 3000 --zg_anneal_kl1 6000 --layer_wise_qkv --grad_clip 100.0 \
# --bart_l 2 --aux_l 3 --sem_coeff 100.0

srun python final_disent.py --test_name "Disentanglement2/OWminii_beta0.6.0.1.0.1.01.4" --kl_beta 0.6 --n_latents 4 --data owt\
 --n_epochs 40 --graph NQKV --max_len 40 --kl_beta_zs 0.1 --kl_beta_zg 0.1 --use_bart --batch_size 64 --z_size 768 --kl_th 0.01 \
 --zg_anneal_kl0 7000 --zg_anneal_kl1 14000 --zs_anneal_kl0 14000 --zs_anneal_kl1 21000 --layer_wise_qkv --grad_clip 100.0 \
 --bart_l 2 --aux_l 3 --sem_coeff 100.0