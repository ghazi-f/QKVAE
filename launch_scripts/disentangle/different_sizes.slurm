#!/usr/bin/env bash
#SBATCH --job-name=RegNSA2
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./disentRegNSA2.txt
#SBATCH --error=./disentRegNSA2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ================= FINAL EXP GROUP ==============================
# Experiments on the SNLI Dataset
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#done
#done


# Experiments on the SNLI Dataset
#for BETA in  0.3 0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer
#done
#done


# Experiments on the SNLI Dataset
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEmzsh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEszsh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEmzmh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEszmh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#done
#done

# ================= REBUTTAL EXP GROUP ==============================

# No self-attention experiments on the SNLI Dataset
for BETA in 0.4 #0.4 0.3
do
for ITER in 1 2 3 4 5
do
srun python disentangle_train.py --test_name "Disentanglement/SNLIRegularNoSA_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --no_sa
done
done


# No self-attention experiments on the SNLI Dataset
#for BETA in  0.3 0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIWideNoSA_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer --no_sa
#done
#done


# Experiments on the SNLI Dataset with 16 LV
#for BETA in  0.3 0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIWider_beta$BETA.$ITER" --kl_beta $BETA --n_latents 16 --data nli --n_epochs 20 --graph IndepInfer
#done
#done

# Experiments on the SNLI Dataset with 32 LV
#for BETA in  0.4 #0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIWidest_beta$BETA.$ITER" --kl_beta $BETA --n_latents 32 --data nli --n_epochs 20 --graph IndepInfer
#done
#done


# Experiments on the SNLI Dataset with 2 LV
#for BETA in  0.3 0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLINarrow_beta$BETA.$ITER" --kl_beta $BETA --n_latents 2 --data nli --n_epochs 20 --graph IndepInfer
#done
#done


# Experiments on the SNLI Dataset with Vanilla Transformer Baseline
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaTr16_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph VanillaTr --z_size 32
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaTr32_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph VanillaTr --z_size 16
#done
#done


# Experiments on the Yelp Dataset with Vanilla Transformer Baseline
#for BETA in 0.3 # 0.4 0.3
#do
#for ITER in 1 #1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/YelpVanillaTr16_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph VanillaTr --z_size 32
#srun python disentangle_train.py --test_name "Disentanglement/YelpVanillaTr32_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph VanillaTr --z_size 16
#done
#done


# ================= REG YELP EXP GROUP ==============================
# Experiments on the SNLI Dataset # Trying with the new Yelp Data
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/YELP2Regular_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data yelp_reg --n_epochs 20 --graph IndepInfer
#done
#done


# Experiments on the SNLI Dataset
#for BETA in  0.3 0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/YELP2Wide_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data yelp_reg --n_epochs 20 --graph IndepInfer
#done
#done


# Experiments on the SNLI Dataset
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/YELP2VanillaVAEmzsh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp_reg --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/YELP2VanillaVAEszsh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp_reg --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/YELP2VanillaVAEmzmh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp_reg --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/YELP2VanillaVAEszmh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp_reg --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#done
#done

# ================= EXP GROUP 1 ==============================
# Experiments on the SNLI Dataset
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#done
#done
#
## Experiments on the Yelp Dataset
#for BETA in 0.3 0.5 #0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/YelpRegular_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer
#done
#done


# ================= EXP GROUP 2 ==============================
# Experiments on the SNLI Dataset
#for BETA in 0.5 #0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEmzsh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEszsh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEmzmh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/SNLIVanillaVAEszmh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#done
#done
#
## ================= EXP GROUP 3 ==============================
## Experiments on the Yelp Dataset
#for BETA in 0.3 0.5 #0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/YelpVanillaVAEmzsh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/YelpVanillaVAEszsh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/YelpVanillaVAEmzmh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/YelpVanillaVAEszmh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#done
#done
#
## ================= EXP GROUP 4 ==============================
## Experiments on the SNLI Dataset
#for BETA in   0.4 0.3 #0.5
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer
##srun python disentangle_train.py --test_name "Disentanglement/SNLIDeep3_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
##srun python disentangle_train.py --test_name "Disentanglement/SNLIDeep2_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
#done
#done
## Experiments on the Yelp Dataset
#for BETA in 0.4 0.3 0.5
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/YelpWide_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data yelp --n_epochs 20 --graph IndepInfer
##srun python disentangle_train.py --test_name "Disentanglement/YelpDeep3_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 4 --data yelp --n_epochs 20 --graph IndepInfer
##srun python disentangle_train.py --test_name "Disentanglement/YelpDeep2_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 4 4 --data yelp --n_epochs 20 --graph IndepInfer
#done
#done

## ================= EXP GROUP 5 ==============================
# Experiments on the SNLI Dataset
#for BETA in   0.3 0.4 #0.5
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIDeep2_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
#done
#done
# Experiments on the SNLI Dataset
#for BETA in   0.4 0.3 #0.5
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train.py --test_name "Disentanglement/SNLIDeep3_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#done
#done

