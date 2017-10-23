Conflict Id: 1, gen: rcnn (modified by me), enc: rcnn, trainable: both, initialize: rand, [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1.0, 0005 not that important)] script: run_conflict1.py, code: conflict1.py


Conflict Id: 2, gen: rcnn (modified by me), enc: rcnn, trainable: both, initialize_enc: [full_enc file](https://github.com/rizwan09/budget_model/blob/dev/model_sparsity_0.0003_coherent_1_dropout_0.1_lr_0.005_max_epochs_100.txt.pkl.gz) , [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1.0, 0005 not that important)] script: run_conflict1.py (edited _type = /CONFLICT2, gpu1, add load_model to the full enc and load_emb = 1), code: conflict1.py (no change will train both)


Conflict Id: 3, gen: rcnn (modified by me), enc: rcnn, trainable: only enc (gen is fixed & loaded from --load_model ../JUST_OUTPUT_LAYER/FIX_ENC_TRAIN_RCNN_GEN2/MODELS/ model from the list of models here by parameters), initialize: rand, [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1 (not 1.0), 0005 not that important)] script: run_conflict3.py, code: conflict3.py



Conflict Id: 4, gen: rcnn (but sentence level), enc = rcnn trainable: only geneartor (enc fixed  [full_enc file](https://github.com/rizwan09/budget_model/blob/dev/model_sparsity_0.0003_coherent_1_dropout_0.1_lr_0.005_max_epochs_100.txt.pkl.gz) )script: run_conflict4.py, code: conflict4.py


Conflict5: genearte a test corpus where senetences are removed if the gold rationales in that sentence is lower than p
run: python conflict5.py --load_rationale ../annotations.json --embedding ../word_vec.gz --aspect 1 --p 0.2 (min gold rationale length /sentence length) (also run_conflict6 has this line]

Conflict6: run full enc on annotations0.1, annotations0.2 etc and record the performance. 

Conflict7: Train full enc on a corpus of originals + blank out] (santy check with sparsity_0.0003_coherent_1_dropout_0.1_lr_0.005)
