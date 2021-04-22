#!/usr/bin/env bash
#SBATCH --job-name=DisOnto
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --cpus-per-task=5
#SBATCH --output=./disentangle_onto.txt
#SBATCH --error=./disentangle_onto.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Unweighted experiments
srun python disentangle_train.py --test_name "Disentanglement/Regular1" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular2" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular3" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular4" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular5" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv21" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv22" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv23" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv24" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv25" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv31" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv32" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv33" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv34" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Deeplv35" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
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

