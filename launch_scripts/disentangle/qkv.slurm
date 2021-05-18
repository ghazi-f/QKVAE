#!/usr/bin/env bash
#SBATCH --job-name=DisOnto
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVYelp.txt
#SBATCH --error=./QKVYelp.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# ================= EXP GROUP 1 ==============================
# Experiments on the SNLI Dataset
#for BETA in 0.5 #0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python qkv_train.py --test_name "Disentanglement/SNLIQKV_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph QKV
#done
#done
#
# Experiments on the Yelp Dataset
for BETA in 0.3 0.5 #0.4
do
for ITER in 1 2 3 4 5
do
srun python qkv_train.py --test_name "Disentanglement/YelpQKV_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data yelp --n_epochs 20 --graph QKV
done
done