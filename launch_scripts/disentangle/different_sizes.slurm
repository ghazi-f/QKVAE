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
# srun python disentangle_train.py --test_name "Disentanglement/low_variables1" --n_latents 8 8 8 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/low_variables2" --n_latents 8 8 8 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/low_variables3" --n_latents 8 8 8 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/low_variables4" --n_latents 8 8 8 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/low_variables5" --n_latents 8 8 8 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/high_variables1" --n_latents 32 32 32 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/high_variables2" --n_latents 32 32 32 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/high_variables3" --n_latents 32 32 32 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/high_variables4" --n_latents 32 32 32 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/high_variables5" --n_latents 32 32 32 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/deep_variables1" --n_latents 16 16 16 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/deep_variables2" --n_latents 16 16 16 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/deep_variables3" --n_latents 16 16 16 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/deep_variables4" --n_latents 16 16 16 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/deep_variables5" --n_latents 16 16 16 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/shallow_variables1" --n_latents 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/shallow_variables2" --n_latents 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/shallow_variables3" --n_latents 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/shallow_variables4" --n_latents 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/shallow_variables5" --n_latents 16 16 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/minimal_enc1" --minimal_enc --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/minimal_enc2" --minimal_enc --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/minimal_enc3" --minimal_enc --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/minimal_enc4" --minimal_enc --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/minimal_enc5" --minimal_enc --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/simple_prior1" --graph NormalSimplePrior --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/simple_prior2" --graph NormalSimplePrior --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/simple_prior3" --graph NormalSimplePrior --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/simple_prior4" --graph NormalSimplePrior --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/simple_prior5" --graph NormalSimplePrior --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/con_gen1" --graph NormalConGen --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/con_gen2" --graph NormalConGen --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/con_gen3" --graph NormalConGen --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/con_gen4" --graph NormalConGen --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/con_gen5" --graph NormalConGen --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/softloss1" --max_elbo_choice 1 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/softloss2" --max_elbo_choice 1 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/softloss3" --max_elbo_choice 1 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/softloss4" --max_elbo_choice 1 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/softloss5" --max_elbo_choice 1 --n_epochs 40
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesbig1" --n_latents 8 8 8 --data ontonotes --n_epochs 40 --graph NormalConGen --max_elbo2 3.0 --encoder_l 3
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesbig2" --n_latents 8 8 8 --data ontonotes --n_epochs 40 --graph NormalConGen --max_elbo2 3.0 --encoder_l 3
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesbig3" --n_latents 8 8 8 --data ontonotes --n_epochs 40 --graph NormalConGen --max_elbo2 3.0 --encoder_l 3
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesbig4" --n_latents 8 8 8 --data ontonotes --n_epochs 40 --graph NormalConGen --max_elbo2 3.0 --encoder_l 3
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesbig5" --n_latents 8 8 8 --data ontonotes --n_epochs 40 --graph NormalConGen --max_elbo2 3.0 --encoder_l 3
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeep1" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeep2" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeep3" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeep4" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeep5" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeper1" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4 --decoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeper2" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4 --decoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeper3" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4 --decoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeper4" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4 --decoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeper5" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0 --encoder_l 4 --decoder_l 4
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslowmax1" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 2.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslowmax2" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 2.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslowmax3" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 2.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslowmax4" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 2.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslowmax5" --n_latents 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 2.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeplv1" --n_latents 8 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeplv2" --n_latents 8 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeplv3" --n_latents 8 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeplv4" --n_latents 8 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonotesdeeplv5" --n_latents 8 8 8 8 --data ontonotes --n_epochs 60 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslongtrain1" --n_latents 8 8 8 --data ontonotes --n_epochs 80 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslongtrain2" --n_latents 8 8 8 --data ontonotes --n_epochs 80 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslongtrain3" --n_latents 8 8 8 --data ontonotes --n_epochs 80 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslongtrain4" --n_latents 8 8 8 --data ontonotes --n_epochs 80 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/ontonoteslongtrain5" --n_latents 8 8 8 --data ontonotes --n_epochs 80 --graph NormalConGen --max_elbo2 3.0
# srun python disentangle_train.py --test_name "Disentanglement/yelp1" --n_latents 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelp2" --n_latents 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelp3" --n_latents 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelp4" --n_latents 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelp5" --n_latents 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpHigh1" --n_latents 8  --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpHigh2" --n_latents 8  --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpHigh3" --n_latents 8  --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpHigh4" --n_latents 8  --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpHigh5" --n_latents 8  --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpDeep1" --n_latents 4 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpDeep2" --n_latents 4 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpDeep3" --n_latents 4 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpDeep4" --n_latents 4 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpDeep5" --n_latents 4 4 --data yelp --n_epochs 60
# srun python disentangle_train.py --test_name "Disentanglement/yelpHighMaxElbo1" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 6.0 --max_elbo2 6.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpHighMaxElbo2" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 6.0 --max_elbo2 6.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpHighMaxElbo3" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 6.0 --max_elbo2 6.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpHighMaxElbo4" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 6.0 --max_elbo2 6.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpHighMaxElbo5" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 6.0 --max_elbo2 6.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpLowMaxElbo1" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 4.0 --max_elbo2 4.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpLowMaxElbo2" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 4.0 --max_elbo2 4.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpLowMaxElbo3" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 4.0 --max_elbo2 4.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpLowMaxElbo4" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 4.0 --max_elbo2 4.0
# srun python disentangle_train.py --test_name "Disentanglement/yelpLowMaxElbo5" --n_latents 4 --data yelp --n_epochs 60 --max_elbo1 4.0 --max_elbo2 4.0
srun python disentangle_train.py --test_name "Disentanglement/shallowIndep1" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/shallowIndep2" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/shallowIndep3" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/shallowIndep4" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/shallowIndep5" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
# srun python disentangle_train.py --test_name "Disentanglement/DeepIndep1" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
# srun python disentangle_train.py --test_name "Disentanglement/DeepIndep2" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
# srun python disentangle_train.py --test_name "Disentanglement/DeepIndep3" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
# srun python disentangle_train.py --test_name "Disentanglement/DeepIndep4" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
# srun python disentangle_train.py --test_name "Disentanglement/DeepIndep5" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer
# srun python disentangle_train.py --test_name "Disentanglement/DeepMaxIndep1" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer --max_elbo_choice 5
# srun python disentangle_train.py --test_name "Disentanglement/DeepMaxIndep2" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer --max_elbo_choice 5
# srun python disentangle_train.py --test_name "Disentanglement/DeepMaxIndep3" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer --max_elbo_choice 5
# srun python disentangle_train.py --test_name "Disentanglement/DeepMaxIndep4" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer --max_elbo_choice 5
# srun python disentangle_train.py --test_name "Disentanglement/DeepMaxIndep5" --n_latents 4 4 4 --data nli --n_epochs 20 --graph IndepInfer --max_elbo_choice 5
srun python disentangle_train.py --test_name "Disentanglement/Regular1" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular2" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular3" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular4" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
srun python disentangle_train.py --test_name "Disentanglement/Regular5" --n_latents 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deep1" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deep2" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deep3" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deep4" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
#srun python disentangle_train.py --test_name "Disentanglement/Deep5" --n_latents 4 4 --data nli --n_epochs 20 --graph IndepInfer
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
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh1" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh2" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh3" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh4" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzsh5" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh1" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh2" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh3" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh4" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEmzmh5" --data nli --n_epochs 20 --graph Vanilla --z_size 32 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh1" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh2" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh3" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh4" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszsh5" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 256 --decoder_h 256
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh1" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh2" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh3" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh4" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512
srun python disentangle_train.py --test_name "Disentanglement/VanillaVAEszmh5" --data nli --n_epochs 20 --graph Vanilla --z_size 16 --encoder_h 512 --decoder_h 512

