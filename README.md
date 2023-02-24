# QKVAE : A model for Unsupervised Disentanglement of Syntax and Semantics
This repo contains the code for our paper [Exploiting Inductive Bias in Transformers for Unsupervised Disentanglement
 of Syntax and Semantics with VAEs](https://aclanthology.org/2022.naacl-main.423/)
 
 The ParaNMT data files we use can be found in the [Google Drive hosted by VGVAE Authors](https://drive.google.com/drive/folders/1HHDlUT_-WpedL6zNYpcN94cLwed_yyrP), except for two files which are [here](https://drive.google.com/drive/folders/1o5r8UBu8efXgN5vPjb3p622CJZDFPfwD?usp=sharing).
 The content of both Google Drives must be downloaded and placed in ```.data/paranmt/```.
 
 After adding the data and installing the dependencies listed in ``requirements.txt``,
 the model can be trained by running:
 ```bash
python qkv_train.py
```
A checkpoint corresponding to the best model we obtained among the 5 instances we ran 
in our paper can be found [in this google drive](https://drive.google.com/file/d/1BbQQP7gzJOQxqRNliuTg87TdR-r6s74H/view?usp=sharing). Note that using this checkpoint requires putting it in a directory named ```checkpoints``` under the root of this repo, and setting the run name with the CLI option ```--test_name QKVBest```.

 
 
 ### To Be Done:
 - Provide usage instructions for syntactic/semantic transfer with the provided
   checkpoint.
 - Package and add evaluation code.
