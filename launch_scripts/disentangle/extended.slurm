#!/usr/bin/env bash
#SBATCH --job-name=SupADVAE2
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./SupADVAE2.txt
#SBATCH --error=./SupADVAE2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ================= ReEvaluating SNLI Exps ==============================
# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEval.csv
#done
#done


# Experiments on the SNLI Dataset with Vanilla Transformer Baseline
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIVanillaTr16_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph VanillaTr --z_size 16 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIVanillaTr32_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph VanillaTr --z_size 32 --csv_out disent_reEval.csv
#done
#done


# Experiments on the SNLI Dataset with Sequence VAE
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIVanillaVAEmzsh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIVanillaVAEszsh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIVanillaVAEmzmh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIVanillaVAEszmh1_beta$BETA.$ITER" --kl_beta $BETA --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512 --csv_out disent_reEval.csv
#done
#done
# ================= ReEvaluating Yelp Exps ==============================

# Experiments on the Yelp Dataset with ADVAE
#for BETA in 0.3 # 0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegular_beta$BETA.$ITER" --kl_beta $BETA --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpWide_beta$BETA.$ITER" --kl_beta $BETA --n_latents 8 --data yelp --n_epochs 20 --graph IndepInfer --csv_out disent_reEval.csv
#done
#done


# Experiments on the Yelp Dataset with Vanilla Transformer Baseline
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpVanillaTr16_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph VanillaTr --z_size 16 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpVanillaTr32_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph VanillaTr --z_size 32 --csv_out disent_reEval.csv
#done
#done


# Experiments on the Yelp Dataset with Sequence VAE
#for BETA in 0.3 #0.4
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpVanillaVAEmzsh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpVanillaVAEszsh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpVanillaVAEmzmh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512 --csv_out disent_reEval.csv
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpVanillaVAEszmh1_beta$BETA.$ITER" --kl_beta $BETA --data yelp --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512 --csv_out disent_reEval.csv
#done
#done


# ================= Trying KL per lv loss ==============================

# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER.LVKL0.1" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvallvkl.csv --lv_kl_coeff 0.1
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER.LVKL0.1" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvallvkl.csv --lv_kl_coeff 0.1
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER.LVKL1.0" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvallvkl.csv --lv_kl_coeff 1.0
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER.LVKL1.0" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvallvkl.csv --lv_kl_coeff 1.0
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER.LVKL100.0" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvallvkl.csv --lv_kl_coeff 100.0
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER.LVKL100.0" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvallvkl.csv --lv_kl_coeff 100.0
#done
#done



# ================= Other N_heads ==============================

# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER.H1" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvalheads.csv --n_heads 1
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER.H2" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvalheads.csv --n_heads 1
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.4 0.3
#do
#for ITER in 1 2 3 4 5
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$ITER.H2" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvalheads.csv --n_heads 2
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIWide_beta$BETA.$ITER.H2" --kl_beta $BETA --n_latents 8 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_reEvalheads.csv --n_heads 2
#done
#done
# ================= Supervised ==============================

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 1.0 10.0 100.0
#do
#for BETA in 0.4
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP
#done
#done
#done

# Experiments on the SNLI Dataset with ADVAE
for SUP in 1.0 10.0 100.0
do
for BETA in 0.3
do
for ITER in 1 2 3
do
srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegular_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP
done
done
done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 1.0 10.0 100.0
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegular_beta0.3.$SUP.$ITER" --kl_beta 0.3 --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP
#done
#done

# Experiments on the Yelp Dataset with ADVAE
#for SUP in 1.0 10.0 100.0
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegular_beta0.4.$SUP.$ITER" --kl_beta 0.4 --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP
#done
#done