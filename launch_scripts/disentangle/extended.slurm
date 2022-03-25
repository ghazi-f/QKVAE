#!/usr/bin/env bash
#SBATCH --job-name=ADVAEBig1
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./ADVAEBig1.txt
#SBATCH --error=./ADVAEBig1.err
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
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularMult_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP  --tr_enc_in_dec
#done
#done
#done

# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularMult_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP --tr_enc_in_dec
#done
#done
#done

# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP --sup_loss_choice single  --tr_enc_in_dec
#done
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP --sup_loss_choice single  --tr_enc_in_dec
#done
#done
#done
# ================= Decoder Supervised ==============================

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularDec_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup2.csv --dec_sup_coeff $SUP  --tr_enc_in_dec
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularDec_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup2.csv --dec_sup_coeff $SUP  --tr_enc_in_dec
#done
#done
#done


# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularDecSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --tr_enc_in_dec --sup_loss_choice single
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularDecSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --tr_enc_in_dec  --sup_loss_choice single
#done
#done
#done


# ================= Unsupervised ==============================

# Experiments on the SNLI Dataset with ADVAE
#for SIZE in 192 768
#do
#for BETA in 0.3 0.4
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularUns2_beta$BETA.$SIZE.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup2.csv --encoder_h $SIZE --decoder_h $SIZE
#done
#done
#done


# Experiments on the SNLI Dataset with ADVAE
#for SIZE in 192 768
#do
#for BETA in 0.3 0.4
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularUnsDeAtt_beta$BETA.$SIZE.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup2.csv --encoder_h $SIZE --decoder_h $SIZE --tr_enc_in_dec
#done
#done
#done

# Experiments on the Yelp Dataset with ADVAE
#for SIZE in 768
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularUns2_beta$BETA.$SIZE.$ITER" --kl_beta $BETA --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --encoder_h $SIZE --decoder_h $SIZE --tr_enc_in_dec
#done
#done
#done

# ================= Wiki ==============================
# Experiments on the Wiki Dataset with ADVAE
#for SIZE in 768
#do
#for BETA in 0.3 0.4
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/WikiRegularUns2_beta$BETA.$SIZE.$ITER" --kl_beta $BETA --n_latents 4 --data wiki --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --encoder_h $SIZE --decoder_h $SIZE --tr_enc_in_dec --max_len 23
#done
#done
#done

# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/WikiRegularMult_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_wiki --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP --tr_enc_in_dec --max_len 23
#done
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 1000.0 #100.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/WikiRegularSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_wiki --n_epochs 20 --graph IndepInfer --csv_out disent_sup.csv --sup_coeff $SUP --sup_loss_choice single --max_len 23 --tr_enc_in_dec
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/WikiRegularDec_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_wiki --n_epochs 20 --graph IndepInfer --csv_out disent_sup2.csv --dec_sup_coeff $SUP  --tr_enc_in_dec --max_len 23
#done
#done
#done


# Experiments on the SNLI Dataset with ADVAE
#for SUP in 1000.0 #100.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/WikiRegularDecSing2_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_wiki --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --tr_enc_in_dec  --max_len 23 --sup_loss_choice single
#done
#done
#done


# ================= Disentanglement with more layers Supervised and Unsupervised ==============================


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/Yelp4RegularSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --sup_coeff $SUP --sup_loss_choice single  --tr_enc_in_dec  --encoder_l 4 --decoder_l 4
#done
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLI4RegularSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --sup_coeff $SUP --sup_loss_choice single  --tr_enc_in_dec  --encoder_l 4 --decoder_l 4
#done
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLI4RegularDecSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --tr_enc_in_dec --sup_loss_choice single  --encoder_l 4 --decoder_l 4
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/Yelp4RegularDecSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --tr_enc_in_dec  --sup_loss_choice single  --encoder_l 4 --decoder_l 4
#done
#done
#done


# Experiments on the SNLI Dataset with ADVAE
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLI4RegularUns_beta$BETA.768.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --tr_enc_in_dec --encoder_l 4 --decoder_l 4
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/Yelp4RegularUns_beta$BETA.768.$ITER" --kl_beta $BETA --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --tr_enc_in_dec --encoder_l 4 --decoder_l 4
#done
#done

# ================= Supervised on both encoder and decoder ==============================

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularBothSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP --sup_coeff $SUP  --tr_enc_in_dec --sup_loss_choice single
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularBothSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --sup_coeff $SUP --tr_enc_in_dec  --sup_loss_choice single
#done
#done
#done

# ================= Supervised with other gamma value ==============================



# Experiments on the SNLI Dataset with ADVAE
#for SUP in 1.0 10.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularDecSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --tr_enc_in_dec --sup_loss_choice single
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 1.0 10.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularDecSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --dec_sup_coeff $SUP  --tr_enc_in_dec  --sup_loss_choice single
#done
#done
#done

# Experiments on the Yelp Dataset with ADVAE
#for SUP in 1.0 10.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --sup_coeff $SUP --sup_loss_choice single  --tr_enc_in_dec
#done
#done
#done

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 1.0 10.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularSing_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 20 --graph IndepInfer --csv_out disent_sup3.csv --sup_coeff $SUP --sup_loss_choice single  --tr_enc_in_dec
#done
#done
#done


# ================= Unsupervised with Auxiliary lv_kl loss==============================
# Experiments on the SNLI Dataset with ADVAE
#for UNSCOEFF in 1.0 10.0 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularUnsKL_beta$BETA.$UNSCOEFF.$ITER" --kl_beta $BETA --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer --lv_kl_coeff $UNSCOEFF --csv_out disent_aux.csv --tr_enc_in_dec --encoder_l 4 --decoder_l 4
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for UNSCOEFF in 1.0 10.0  100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularUnsKL_beta$BETA.$UNSCOEFF.$ITER" --kl_beta $BETA --n_latents 4 --data yelp --n_epochs 20 --graph IndepInfer --lv_kl_coeff $UNSCOEFF --csv_out disent_aux.csv --tr_enc_in_dec --encoder_l 4 --decoder_l 4
#done
#done
#done


# ================= Supervised on both encoder and decoder with biggers Transformers ==============================

# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularBoth4_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 40 --graph IndepInfer --csv_out disent_sup5.csv --dec_sup_coeff $SUP --sup_coeff $SUP  --tr_enc_in_dec --sup_loss_choice single  --anneal_kl0 20000 --anneal_kl1 30000 --encoder_l 4 --decoder_l 4
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
for SUP in 100.0 1000.0
do
for BETA in 0.3
do
for ITER in 1 2 3
do
srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularBoth4_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 40 --graph IndepInfer --csv_out disent_sup5.csv --dec_sup_coeff $SUP  --sup_coeff $SUP --tr_enc_in_dec  --sup_loss_choice single  --anneal_kl0 20000 --anneal_kl1 30000  --encoder_l 4 --decoder_l 4
done
done
done


# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularBoth3K_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 40 --graph IndepInfer --csv_out disent_sup5.csv --dec_sup_coeff $SUP --sup_coeff $SUP  --tr_enc_in_dec --sup_loss_choice single  --anneal_kl0 20000 --anneal_kl1 30000 --encoder_h 3072 --decoder_h 3072
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularBoth3K_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 40 --graph IndepInfer --csv_out disent_sup5.csv --dec_sup_coeff $SUP  --sup_coeff $SUP --tr_enc_in_dec  --sup_loss_choice single  --anneal_kl0 20000 --anneal_kl1 30000  --encoder_h 3072 --decoder_h 3072
#done
#done
#done


# Experiments on the SNLI Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/SNLIRegularBoth12H_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_nli --n_epochs 40 --graph IndepInfer --csv_out disent_sup5.csv --dec_sup_coeff $SUP --sup_coeff $SUP  --tr_enc_in_dec --sup_loss_choice single  --anneal_kl0 20000 --anneal_kl1 30000 --n_heads 12
#done
#done
#done


# Experiments on the Yelp Dataset with ADVAE
#for SUP in 100.0 1000.0
#do
#for BETA in 0.3
#do
#for ITER in 1 2 3
#do
#srun python disentangle_train_ex.py --test_name "Disentanglement/YelpRegularBoth12H_beta$BETA.$SUP.$ITER" --kl_beta $BETA --n_latents 4 --data sup_yelp --n_epochs 40 --graph IndepInfer --csv_out disent_sup5.csv --dec_sup_coeff $SUP  --sup_coeff $SUP --tr_enc_in_dec  --sup_loss_choice single  --anneal_kl0 20000 --anneal_kl1 30000  --n_heads 12
#done
#done
#done
