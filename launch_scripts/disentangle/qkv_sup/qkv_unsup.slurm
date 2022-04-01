#!/usr/bin/env bash
#SBATCH --job-name=QKVUnsID
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVUnsID.txt
#SBATCH --error=./QKVUnsID.err
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH -A jrp@gpu

# Experiments on the SNLI Dataset with ADVAE
for BETA in 0.6
do
for ITER in 1 2 3
do
srun python qkv_train.py --test_name "Disentanglement/SNLIQKVUns_beta$BETA.0.3.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph QKV --csv_out disent_qkv.csv --batch_size 128 --kl_beta_zs 0.3 --z_size 768 --zs_anneal_kl0 7000 --zs_anneal_kl1 11000
done
done


# Experiments on the Yelp Dataset with ADVAE
for BETA in 0.6
do
for ITER in 1 2 3
do
srun python qkv_train.py --test_name "Disentanglement/YelpQKVUns_beta$BETA.0.3.$ITER" --kl_beta $BETA --n_latents 4 --data yelp --n_epochs 20 --graph QKV --csv_out disent_qkv.csv --batch_size 128 --kl_beta_zs 0.3 --z_size 768 --zs_anneal_kl0 7000 --zs_anneal_kl1 11000
done
done
