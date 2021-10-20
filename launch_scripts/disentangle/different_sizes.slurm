#!/usr/bin/env bash
#SBATCH --job-name=SNLIReg
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./disentNLIReg.txt
#SBATCH --error=./disentNLIReg.err
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

# ================= REG YELP EXP GROUP ==============================
# Experiments on the SNLI Dataset # Trying with the new Yelp Data
for BETA in 0.4 0.3
do
for ITER in 1 2 3 4 5
do
srun python disentangle_train.py --test_name "Disentanglement/YELP2Regular_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data yelp_reg --n_epochs 20 --graph IndepInfer
done
done


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


#srun python disentangle_train.py --test_name "Disentanglement/Regular1" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Regular2" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Regular3" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Regular4" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Regular5" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv21" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv22" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv23" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv24" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv25" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv31" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv32" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv33" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv34" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deeplv35" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/DeepNet1" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --encoder_l 3 --decoder_l 3
#srun python disentangle_train.py --test_name "Disentanglement/DeepNet2" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --encoder_l 3 --decoder_l 3
#srun python disentangle_train.py --test_name "Disentanglement/DeepNet3" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --encoder_l 3 --decoder_l 3
#srun python disentangle_train.py --test_name "Disentanglement/DeepNet4" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --encoder_l 3 --decoder_l 3
#srun python disentangle_train.py --test_name "Disentanglement/DeepNet5" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --encoder_l 3 --decoder_l 3
#srun python disentangle_train.py --test_name "Disentanglement/Wide1" --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Wide2" --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Wide3" --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Wide4" --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Wide5" --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/YelpRegular1" --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/YelpRegular2" --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/YelpRegular3" --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/YelpRegular4" --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/YelpRegular5" --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh1" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh2" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh3" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh4" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh5" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh1" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh2" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh3" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh4" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh5" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh1" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh2" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh3" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh4" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh5" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh1" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh2" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh3" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh4" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
#srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh5" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512

