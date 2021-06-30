#!/usr/bin/env bash
#SBATCH --job-name=QKVYelp
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVYelp.txt
#SBATCH --error=./QKVYelp.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# ================= EXP GROUP 1 ==============================
# Experiments on the SNLI Dataset
#for NL in 4 8
#do
#for K in 8 16
#do
#for BETA in 0.5 0.7
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/SNLIQKV_beta$BETA.$ITER.$K.$NL" --kl_beta $BETA --n_latents $NL --data nli\
# --n_epochs 20 --graph QKV --max_elbo_choice 6 --n_keys $K
#done
#done
#done
#done

# Experiments on the Yelp Dataset
#for NL in 4 8
#do
#for K in 4 #8 16
#do
#for BETA in 0.3 0.4 #0.5 0.7
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/YelpQKV_beta$BETA.$ITER.$K.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
# --n_epochs 20 --graph QKV --max_elbo_choice 6 --n_keys $K
#done
#done
#done
#done


## Experiments on the Ontonotes Dataset
#for NL in 4 8
#do
#for K in 8 16
#do
#for BETA in 0.5 0.7
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/OntoQKV_beta$BETA.$ITER.$K.$NL" --kl_beta $BETA --n_latents $NL --data ontonotes\
# --n_epochs 20 --graph QKV --max_elbo_choice 6 --n_keys $K
#done
#done
#done
#done


## Experiments on the ParaNMT Dataset
# for NL in 8
# do
# for K in 4
# do
# for BETA in 0.5
# do
# for ITER in 1 2 3
# do
# srun python qkv_train.py --test_name "Disentanglement/ParaQKV_beta$BETA.$ITER.$K.$NL" --kl_beta $BETA --n_latents $NL --data paranmt\
#  --n_epochs 5 --graph HQKV --max_elbo_choice 6 --n_keys $K --max_len 40
# done
# done
# done
# done

## Experiments on the ParaNMT Dataset 2
srun python qkv_train.py --test_name "Disentanglement/ParaQKV_Reg" --kl_beta 0.3 --n_latents 4 \
--data paranmt --n_epochs 10 --graph QKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid
# srun python qkv_train.py --test_name "Disentanglement/ParaQKV_Big" --kl_beta 0.3 --n_latents 4 \
# --data paranmt --n_epochs 10 --graph QKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid\
# --z_emb_dim 1536 --encoder_h 1536 --decoder_h 1536
# srun python qkv_train.py --test_name "Disentanglement/ParaQKV_klw" --kl_beta 0.5 --n_latents 4 --kl_beta_zs 0.2\
# --data paranmt --n_epochs 10 --graph QKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid
# srun python qkv_train.py --test_name "Disentanglement/ParaQKV_sgd" --kl_beta 0.3 --n_latents 4 \
# --data paranmt --n_epochs 10 --graph QKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer sgd --anneal_kl_type sigmoid
# srun python qkv_train.py --test_name "Disentanglement/ParaQKV_hqkv" --kl_beta 0.3 --n_latents 4 \
# --data paranmt --n_epochs 10 --graph HQKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid
# srun python qkv_train.py --test_name "Disentanglement/ParaQKV_linear" --kl_beta 0.3 --n_latents 4 \
# --data paranmt --n_epochs 10 --graph HQKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type linear