#!/usr/bin/env bash
#SBATCH --job-name=QKVSNLI
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVSNLI.txt
#SBATCH --error=./QKVSNLI.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# ================= EXP GROUP 1 ==============================
## Experiments on the SNLI Dataset
for NL in 4 8
for K in 8 16
for BETA in 0.5 0.7
do
for ITER in 1 2 3
do
srun python qkv_train.py --test_name "Disentanglement/SNLIQKV_beta$BETA.$ITER.$K.$NL" --kl_beta $BETA --n_latents $NL --data nli\
 --n_epochs 20 --graph QKV --max_elbo_choice 6 --n_keys $K
done
done
done
done

#for NL in 4 8
#for K in 8 16
# Experiments on the Yelp Dataset
#for BETA in 0.5 0.7
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/YelpQKV_beta$BETA.$ITER.$K.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
# --n_epochs 20 --graph QKV --max_elbo_choice 6 --n_keys $K
#done
#done
#done
#done