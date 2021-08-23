#!/usr/bin/env bash
#SBATCH --job-name=HQKVNMinLen
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVParaMinLen4.txt
#SBATCH --error=./QKVParaMinLen4.err
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

#srun python qkv_train.py --test_name "Disentanglement/ParaQKV_Reg2" --kl_beta 0.4 --kl_beta_zs 0.01 --n_latents 4 \
#--data paranmt --n_epochs 10 --graph QKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid
#srun python qkv_train.py --test_name "Disentanglement/ParaQKV_hqkv2" --kl_beta 0.4 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 4 \
#--data paranmt --n_epochs 10 --graph HQKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid
#srun python qkv_train.py --test_name "Disentanglement/ParaQKV_Reg3" --kl_beta 0.3 --kl_beta_zs 0.01 --n_latents 4 \
#--data paranmt --n_epochs 10 --graph QKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid
#srun python qkv_train.py --test_name "Disentanglement/ParaQKV_hqkv3" --kl_beta 0.3 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 4 \
#--data paranmt --n_epochs 10 --graph HQKV --max_elbo_choice 6 --n_keys 4 --max_len 20 --optimizer adam --anneal_kl_type sigmoid


#for BETA in 0.5 0.6 0.7
#do
#srun python qkv_train.py --test_name "Disentanglement/ParaHQKVBeta$BETA" --kl_beta $BETA --kl_beta_zs 0.01 --kl_beta_zg 0.1 \
#--n_latents 4 --data paranmt --n_epochs 10 --graph HQKV --max_elbo_choice 6 --n_keys 8 --max_len 20 --anneal_kl_type sigmoid
#done


#for NL in 8 16 32
#do
#srun python qkv_train.py --test_name "Disentanglement/ParaHQKVNL$NL" --kl_beta 0.5 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents $NL \
#--data paranmt --n_epochs 10 --graph HQKV --max_elbo_choice 6 --n_keys 8 --max_len 20 --anneal_kl_type sigmoid
#done


# for NK in 8 16 32
# do
# srun python qkv_train.py --test_name "Disentanglement/ParaHQKVNK$NK" --kl_beta 0.5 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 4 \
# --data paranmt --n_epochs 10 --graph HQKV --max_elbo_choice 6 --n_keys $NK --max_len 20 --anneal_kl_type sigmoid
# done

#srun python qkv_train.py --test_name "Disentanglement/ParaHQKVN8" --kl_beta 0.5 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 8 \
#--data paranmt --n_epochs 20 --graph HQKV --max_elbo_choice 6 --n_keys 8 --max_len 20 --anneal_kl_type sigmoid
#done


#srun python qkv_train.py --test_name "Disentanglement/ParaHQKVN16" --kl_beta 0.5 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 16 \
#--data paranmt --n_epochs 20 --graph HQKV --max_elbo_choice 6 --n_keys 16 --max_len 20 --anneal_kl_type sigmoid
#done

# srun python qkv_train.py --test_name "Disentanglement/ParaHQKVN32" --kl_beta 0.5 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 32 \
# --data paranmt --n_epochs 20 --graph HQKV --max_elbo_choice 6 --n_keys 32 --max_len 20 --anneal_kl_type sigmoid
# done

# ========= Limited Sentence size experiments on ParaNMT =========

# srun python qkv_train.py --test_name "Disentanglement/ParaMinLen1" --kl_beta 0.3 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 16 \
# --data paranmt --n_epochs 20 --graph HQKV --max_elbo_choice 6 --n_keys 16 --max_len 20 --anneal_kl_type sigmoid
# done

# srun python qkv_train.py --test_name "Disentanglement/ParaMinLen2" --kl_beta 0.4 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 16 \
# --data paranmt --n_epochs 20 --graph HQKV --max_elbo_choice 6 --n_keys 16 --max_len 20 --anneal_kl_type sigmoid
# done

# srun python qkv_train.py --test_name "Disentanglement/ParaMinLen3" --kl_beta 0.5 --kl_beta_zs 0.01 --kl_beta_zg 0.1 --n_latents 16 \
# --data paranmt --n_epochs 20 --graph HQKV --max_elbo_choice 6 --n_keys 16 --max_len 20 --anneal_kl_type sigmoid
# done

#srun python qkv_train.py --test_name "Disentanglement/ParaMinLen4" --kl_beta 0.5 --kl_beta_zs 0.01 --kl_beta_zg 0.01 --n_latents 16 \
#--data paranmt --n_epochs 20 --graph HQKV --max_elbo_choice 6 --n_keys 16 --max_len 20 --anneal_kl_type sigmoid
#done


# ========= Pre-Final Launches =========

#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/ParaQKV_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data paranmt\
# --n_epochs 5 --graph QKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs 0.01 --kl_beta_zg 0.1
#done
#done
#done


#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/YelpQKV_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
#--n_epochs 20 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.01 --kl_beta_zg 0.1
#done
#done
#done


#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/ParaQKV2_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data paranmt\
# --n_epochs 5 --graph QKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs 0.1 --kl_beta_zg 0.1
#done
#done
#done


#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/YelpQKV2_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
#--n_epochs 20 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.1 --kl_beta_zg 0.1
#done
#done
#done


#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/ParaHQKV_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data paranmt\
# --n_epochs 5 --graph HQKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs 0.01 --kl_beta_zg 0.1
#done
#done
#done


#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/YelpHQKV_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
#--n_epochs 20 --graph HQKV --max_elbo_choice 6 --kl_beta_zs 0.01 --kl_beta_zg 0.1
#done
#done
#done


#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/ParaDisc_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data paranmt\
# --n_epochs 5 --graph HQKVDiscZs --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs 0.01 --kl_beta_zg 0.1
#done
#done
#done


#for NL in 4 8 16
#do
#for BETA in 0.3 0.4 0.5
#do
#for ITER in 1 2 3
#do
#srun python qkv_train.py --test_name "Disentanglement/YelpDisc_beta$BETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
#--n_epochs 20 --graph HQKVDiscZs --max_elbo_choice 6 --kl_beta_zs 0.01 --kl_beta_zg 0.1
#done
#done
#done