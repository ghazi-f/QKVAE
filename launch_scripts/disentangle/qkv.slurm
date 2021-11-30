#!/usr/bin/env bash
#SBATCH --job-name=QKVReg45
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./QKVReg45.txt
#SBATCH --error=./QKVReg45.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
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

#for ITER in 1 2 3
#do
#for NL in 4 8 16
#do
#for BETA in 0.6 0.7 #0.5
#do
#for ZSBETA in 0.1 0.3
#do
#srun python qkv_train.py --test_name "Disentanglement2/ParaQKV_beta$BETA.$ZSBETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data paranmt\
# --n_epochs 20 --graph QKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs $ZSBETA --kl_beta_zg 0.1
#done
#done
#done
#done


#for ITER in 1 2 3
#do
#for NL in 4 8 16
#do
#for BETA in 0.6 0.7 #0.5
#do
#for ZSBETA in 0.1 0.3
#do
#srun python qkv_train.py --test_name "Disentanglement2/YelpQKV_beta$BETA.$ZSBETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
# --n_epochs 20 --graph QKV --max_elbo_choice 6 --kl_beta_zs $ZSBETA --kl_beta_zg 0.1
#done
#done
#done
#done



#for ITER in 1 2 3
#do
#for NL in 4 8 16
#do
#for BETA in 0.6 0.7 #
#do
#for ZSBETA in 0.1 0.3
#do
#srun python qkv_train.py --test_name "Disentanglement2/ParaHQKV_beta$BETA.$ZSBETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data paranmt\
# --n_epochs 20 --graph HQKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs $ZSBETA --kl_beta_zg 0.1
#done
#done
#done
#done


#for ITER in 1 2 3
#do
#for NL in 4 8 16
#do
#for BETA in 0.6 0.7 #0.5
#do
#for ZSBETA in 0.1 0.3
#do
#srun python qkv_train.py --test_name "Disentanglement2/YelpHQKV_beta$BETA.$ZSBETA.$ITER.$NL" --kl_beta $BETA --n_latents $NL --data yelp\
# --n_epochs 20 --graph HQKV --max_elbo_choice 6 --kl_beta_zs $ZSBETA --kl_beta_zg 0.1
#done
#done
#done
#done

# ========= Long Launches =========
# Para
#srun python qkv_train.py --test_name "Disentanglement2/ParaHQKV_beta0.6.0.3.1.16" --kl_beta 0.6 --n_latents 16 --data paranmt\
# --n_epochs 200 --graph HQKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs 0.3 --kl_beta_zg 0.1

#srun python qkv_train.py --test_name "Disentanglement2/ParaQKV_beta0.6.0.3.1.16" --kl_beta 0.6 --n_latents 16 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs 0.3 --kl_beta_zg 0.1

#srun python qkv_train.py --test_name "Disentanglement2/ParaQKV_beta0.6.0.3.1.32" --kl_beta 0.6 --n_latents 32 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --init_len 12 --kl_beta_zs 0.3 --kl_beta_zg 0.1

# Yelp
#srun python qkv_train.py --test_name "Disentanglement2/YelpHQKV_beta0.6.0.3.1.16" --kl_beta 0.6 --n_latents 16 --data yelp\
# --n_epochs 200 --graph HQKV --max_elbo_choice 6 --kl_beta_zs 0.3 --kl_beta_zg 0.1

#srun python qkv_train.py --test_name "Disentanglement2/YelpQKV_beta0.6.0.3.1.16" --kl_beta 0.6 --n_latents 16 --data yelp\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --kl_beta_zg 0.1

#srun python qkv_train.py --test_name "Disentanglement2/YelpQKV_beta0.6.0.3.1.32" --kl_beta 0.6 --n_latents 32 --data yelp\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --kl_beta_zg 0.1

# ========= BART Launches =========
# Para
#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBART_beta0.3.0.3.1.4" --kl_beta 0.3 --n_latents 4 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.3 --use_bart --batch_size 64

#srun python qkv_train.py --test_name "Disentanglement2/ParaHQKVBART_beta0.3.0.1.1.16" --kl_beta 0.3 --n_latents 16 --data paranmt\
# --n_epochs 200 --graph HQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.1 --use_bart --batch_size 64 --kl_beta_zg 0.1
#srun python qkv_train.py --test_name "Disentanglement2/ParaHQKVBART_beta0.3.0.2.1.16" --kl_beta 0.3 --n_latents 16 --data paranmt\
# --n_epochs 200 --graph HQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.2 --use_bart --batch_size 64 --kl_beta_zg 0.2

#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBART_beta0.1.0.1.1.16" --kl_beta 0.1 --n_latents 16 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.1  --use_bart --batch_size 64

#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBART_beta0.4.0.3.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.3 --use_bart --batch_size 64

#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBARTmini_beta0.3.0.3.1.8" --kl_beta 0.3 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.3 --use_bart --batch_size 64 --z_size 192


#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBARTmini_beta0.3.0.3.1.8" --kl_beta 0.3 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.3 --use_bart --batch_size 64 --z_size 192


#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBARTsmaller_beta0.3.0.3.1.8" --kl_beta 0.3 --n_latents 16 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.3 --use_bart --batch_size 64 --z_size 96
#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBARTsmaller_beta0.3.0.3.1.8" --kl_beta 0.3 --n_latents 16 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.3 --use_bart --batch_size 64 --z_size 96

#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBARTminiLag_beta0.6.0.6.1.8" --kl_beta 0.6 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.6 --use_bart --batch_size 64 --z_size 192 --losses LagVAE
#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBARTminiLag_beta0.4.0.4.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --losses LagVAE
#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBARTminiLag_beta0.6.0.6.1.8" --kl_beta 0.6 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.6 --use_bart --batch_size 64 --z_size 192 --losses LagVAE
#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBARTminiLag_beta0.4.0.4.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --losses LagVAE


#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBARTminiLag_beta0.2.0.2.1.8" --kl_beta 0.2 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.2 --use_bart --batch_size 64 --z_size 192 --losses LagVAE
#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBARTminiLag_beta0.2.0.2.1.8" --kl_beta 0.2 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.2 --use_bart --batch_size 64 --z_size 192 --losses LagVAE


#srun python qkv_train.py --test_name "Disentanglement2/ParaSQKVBARTminiLag_beta0.4.0.4.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph SQKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05
#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBARTminiLag_beta0.4.0.4.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05
#srun python qkv_train.py --test_name "Disentanglement2/ParaQKVBARTminiThr_beta0.2.0.2.1.8" --kl_beta 0.2 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_elbo_choice 6 --max_len 40 --kl_beta_zs 0.2 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05

# Yelp
#srun python qkv_train.py --test_name "Disentanglement2/YelpQKVBART_beta0.3.0.3.1.4" --kl_beta 0.3 --n_latents 4 --data yelp\
#  --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --max_len 40 --use_bart --batch_size 64

#srun python qkv_train.py --test_name "Disentanglement2/YelpHQKVBART_beta0.3.0.1.1.16" --kl_beta 0.3 --n_latents 16 --data yelp\
#  --n_epochs 200 --graph HQKV --max_elbo_choice 6 --kl_beta_zs 0.1 --max_len 40 --use_bart --batch_size 64 --kl_beta_zg 0.1
#srun python qkv_train.py --test_name "Disentanglement2/YelpHQKVBART_beta0.3.0.2.1.16" --kl_beta 0.3 --n_latents 16 --data yelp\
#  --n_epochs 200 --graph HQKV --max_elbo_choice 6 --kl_beta_zs 0.2 --max_len 40 --use_bart --batch_size 64 --kl_beta_zg 0.2

#srun python qkv_train.py --test_name "Disentanglement2/YelpSQKVBART_beta0.1.0.1.1.16" --kl_beta 0.1 --n_latents 16 --data yelp\
#  --n_epochs 200 --graph SQKV --max_elbo_choice 6 --kl_beta_zs 0.1 --max_len 40 --use_bart --batch_size 64

#srun python qkv_train.py --test_name "Disentanglement2/YelpQKVBART_beta0.1.0.1.1.16" --kl_beta 0.1 --n_latents 16 --data yelp\
#  --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.1 --max_len 40 --use_bart --batch_size 64

#srun python qkv_train.py --test_name "Disentanglement2/YelpQKVBARTmini1_beta0.3.0.3.1.16" --kl_beta 0.3 --n_latents 16 --data yelp\
#  --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --max_len 40 --use_bart --batch_size 64 --z_size 368

#srun python qkv_train.py --test_name "Disentanglement2/YelpQKVBARTmini2_beta0.3.0.3.1.16" --kl_beta 0.3 --n_latents 16 --data yelp\
#  --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --max_len 40 --use_bart --batch_size 64 --z_size 192

#srun python qkv_train.py --test_name "Disentanglement2/YelpQKVBARTmini3_beta0.3.0.3.1.16" --kl_beta 0.3 --n_latents 8 --data yelp\
#  --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --max_len 40 --use_bart --batch_size 64 --z_size 192

#srun python qkv_train.py --test_name "Disentanglement2/YelpSQKVBARTmini_beta0.3.0.3.1.16" --kl_beta 0.3 --n_latents 8 --data yelp\
#  --n_epochs 200 --graph SQKV --max_elbo_choice 6 --kl_beta_zs 0.3 --max_len 40 --use_bart --batch_size 64 --z_size 192


#srun python qkv_train.py --test_name "Disentanglement2/YelpQKVBARTmini4_beta0.3.0.3.1.16" --kl_beta 0.3 --n_latents 4 --data yelp\
#  --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --max_len 40 --use_bart --batch_size 64 --z_size 192

#srun python qkv_train.py --test_name "Disentanglement2/YelpQKVBARTmini5_beta0.3.0.3.1.16" --kl_beta 0.3 --n_latents 8 --data yelp\
#  --n_epochs 200 --graph QKV --max_elbo_choice 6 --kl_beta_zs 0.3 --max_len 40 --use_bart --batch_size 64 --z_size 96

#ParaQKVBARTminiThr


#srun python qkv_train.py --test_name "Disentanglement2/PQBmT_beta0.4.0.4.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05


#srun python qkv_train.py --test_name "Disentanglement2/PQBmTGA_beta0.4.0.4.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05 --grad_accu 4

#srun python qkv_train.py --test_name "Disentanglement2/PQBmTLongAnneal_beta0.4.0.4.1.8" --kl_beta 0.4 --n_latents 8 --data paranmt\
# --n_epochs 200 --graph QKV --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05 \
# --anneal_kl0 20000 --anneal_kl1 100000 --zs_anneal_kl0 20000 --zs_anneal_kl1 100000


#srun python qkv_train.py --test_name "Disentanglement2/PQBmT_beta0.4.0.4.1.32" --kl_beta 0.4 --n_latents 32 --data paranmt\
# --n_epochs 200 --graph QKV --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05


#srun python qkv_train.py --test_name "Disentanglement2/PQBmT_beta0.4.0.4.1.4" --kl_beta 0.4 --n_latents 4 --data paranmt\
# --n_epochs 200 --graph QKV --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05


srun python qkv_train.py --test_name "Disentanglement2/PQBmT_beta0.5.0.4.1.4" --kl_beta 0.5 --n_latents 4 --data paranmt\
 --n_epochs 200 --graph QKV --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05


#srun python qkv_train.py --test_name "Disentanglement2/PQBmT_beta0.6.0.4.1.4" --kl_beta 0.6 --n_latents 4 --data paranmt\
# --n_epochs 200 --graph QKV --max_len 40 --kl_beta_zs 0.4 --use_bart --batch_size 64 --z_size 192 --kl_th 0.05



