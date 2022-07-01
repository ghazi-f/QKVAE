# QKVAE : A model for Unsupervised Disentanglement of Syntax and Semantics
This repo contains the code for our paper [Exploiting Inductive Bias in Transformers for Unsupervised Disentanglement
 of Syntax and Semantics with VAEs](https://arxiv.org/abs/2205.05943)
 
 The ParaNMT data files we use can be found in the [Google Drive hosted by VGVAE Authors](https://drive.google.com/drive/folders/1HHDlUT_-WpedL6zNYpcN94cLwed_yyrP), except for two files which are [here](https://drive.google.com/drive/folders/1o5r8UBu8efXgN5vPjb3p622CJZDFPfwD?usp=sharing).
 The content of both Google Drives must be downloaded and placed in ```.data/paranmt/```.
 
 After adding the data and installing the dependencies listed in ``requirements.txt``,
 the model can be trained by running:
 ```bash
python qkv_train.py
```
A checkpoint corresponding to the best model we obtained among the 5 instances we ran 
in our paper can be found [in this google drive](https://drive.google.com/file/d/1LEBovmLN3kkHNE7Z_nk5IACMxhyk5-G4/view?usp=sharing).
 
 
 ### To Be Done:
 - Provide usage instructions for syntactic/semantic transfer with the provided
   checkpoint.
 - Package and add evaluation code.
