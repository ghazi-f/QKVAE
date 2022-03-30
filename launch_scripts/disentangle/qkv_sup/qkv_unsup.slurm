#!/usr/bin/env bash
#SBATCH --job-name=QKVUnsYelp
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVUnsYelp.txt
#SBATCH --error=./QKVUnsYelp.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH -A jrp@gpu

# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/SNLIQKVUns_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph QKV --csv_out disent_qkv.csv --batch_size 128 --kl_beta_zs $BETA --z_size 768 --zs_anneal_kl0 7000 --zs_anneal_kl1 11000
#done
#done


# Experiments on the Yelp Dataset with ADVAE
for BETA in 0.3 0.4 0.5
do
for ITER in 1 2 3
do
srun python qkv_train.py --test_name "Disentanglement/YelpQKVUns_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data yelp --n_epochs 20 --graph QKV --csv_out disent_qkv.csv --batch_size 128 --kl_beta_zs $BETA --z_size 768 --zs_anneal_kl0 7000 --zs_anneal_kl1 11000
done
done
