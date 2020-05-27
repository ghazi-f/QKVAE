""" 23/05/2020
"--max_len", default=32
"--batch_size", default=10
"--grad_accu", default=8
"--n_epochs", default=100
"--test_freq", default=16
"--complete_test_freq", default=80
"--supervision_proportion", default=1
"--unsupervision_proportion", default=1
"--generation_weight", default=1e-2
"--device", default='cuda:0'
"--embedding_dim", default=400
"--pos_embedding_dim", default=50
"--z_size", default=100
"--text_rep_l", default=3
"--text_rep_h", default=200
"--encoder_h", default=200
"--encoder_l", default=1
"--pos_h", default=200
"--pos_l", default=1
"--decoder_h", default=600
"--decoder_l", default=3
"--highway", default=False
"--markovian", default=False
"--losses", default='SSVAE'
"--l2_reg", default=0.0
"--training_iw_samples", default=5
"--testing_iw_samples", default=5
"--test_prior_samples", default=5
"--anneal_kl0", default=800
"--anneal_kl1", default=2400
"--grad_clip", default=10
"--kl_th", default=None, type
"--dropout", default=0.33
"--lr", default=2e-3
"--lr_reduction", default=10
"--wait_epochs", default=5


"""