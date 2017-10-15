Conflict Id: 1, gen: rcnn (modified by me), enc: rcnn, trainable: both, initialize: rand, [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1.0, 0005 not that important)] script: run_conflict1.py, code: conflict1.py


Conflict Id: 2, gen: rcnn (modified by me), enc: rcnn, trainable: both, initialize: [full_enc file](https://github.com/rizwan09/budget_model/blob/dev/model_sparsity_0.0003_coherent_1_dropout_0.1_lr_0.005_max_epochs_100.txt.pkl.gz) , [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1.0, 0005 not that important)] script: run_conflict1.py (edited _type = /CONFLICT2, gpu1, add load_model to the full enc and load_emb = 1), code: conflict1.py (no change will train both)


Conflict Id: 3, gen: rcnn (modified by me), enc: rcnn, trainable: only enc (gen is fixed & loaded from model from the list of models here by parameters), initialize: rand, [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1.0, 0005 not that important)] script: run_conflict3.py, code: conflict3.py
