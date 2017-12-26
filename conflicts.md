Conflict Id: 1, gen: rcnn (modified by me), enc: rcnn, trainable: both, initialize: rand, [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1.0, 0005 not that important)] script: run_conflict1.py, code: conflict1.py


Conflict Id: 2, gen: rcnn (modified by me), enc: rcnn, trainable: both, initialize_enc: [joint full_enc file](https://github.com/rizwan09/budget_model/blob/dev/model_sparsity_0.0003_coherent_1_dropout_0.1_lr_0.005_max_epochs_100.txt.pkl.gz) , [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1.0, 0005 not that important)] script: run_conflict1.py (edited _type = /CONFLICT2, gpu1, add load_model to the full enc and load_emb = 1), code: conflict1.py (no change will train both)


Conflict Id: 3, gen: rcnn (modified by me), enc: rcnn, trainable: only enc (gen is fixed & loaded from --load_model ../JUST_OUTPUT_LAYER/FIX_ENC_TRAIN_RCNN_GEN2/MODELS/ model from the list of models here by parameters), initialize: rand, [special run setting: (lamda1, lamda2, lr):: (0.0003, 0.8, 0005), (0.0001, 0.5, 0005)(0.00012, 1 (not 1.0), 0005 not that important)] script: run_conflict3.py, code: conflict3.py



Conflict Id: 4, gen: rcnn (but sentence level), enc = rcnn trainable: only geneartor (enc fixed [joint full_enc file](https://github.com/rizwan09/budget_model/blob/dev/model_sparsity_0.0003_coherent_1_dropout_0.1_lr_0.005_max_epochs_100.txt.pkl.gz) )script: run_conflict4.py, code: conflict4.py


Conflict5: genearte a test corpus where senetences are removed if the gold rationales in that sentence is lower than p
run: python conflict5.py --load_rationale ../annotations.json --embedding ../word_vec.gz --aspect 1 --p 0.2 (min gold rationale length /sentence length) (also run_conflict6 has this line]

Conflict6: run full enc on annotations0.1, annotations0.2 etc and record the performance. 

#### the folowings are fruitful codes


plot_conflict1 (in folder: graph_data): plot different tables in a same graph. (i.e., first run_conflict10 generates tables for full enc and just_output_layer, linear, rcnn generator, now plot_conflict1 will plot these tables in a same graph)


plot_conflict2 (in folder: graph_data): plot different tables in a same graph. (i.e., first run_conflict25 generates tables for full enc and just_output_layer, rcnn generator, now plot_conflict2 will plot these tables in a same graph)

Conflict7: train data preprocessing to generate a train set where some sentences are blanked out.
to run: THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python conflict7.py --train ../reviews.aspect1.train.txt.gz --max_epochs 5, this code is appended in conflict8

Conflict8: Train full enc on a corpus of originals + blank out (sanity check with sparsity_0.0003_coherent_1_dropout_0.1_lr_0.005) {also to reproduce the full enc [performace](https://docs.google.com/spreadsheets/d/1xQmQpaoojtVGbEJT4CY_qqMzBTnjq_uRZ9vDfjQHVko/edit#gid=0) }

conflict9:  test with set (all gold + randomly selection of sentences depending on args.selection) and record the result to see, the performance of mse vs selection

run_rcnn_gen: I can train jointly rcnn gen + rcnn enc, and later load only the rcnn gen part with the full enc from the other model by running rcnn_gen.py

run_conflict10: load union and original full encoder and (with no generator or with loading defferent geneartor like rcnn of tao lei (conflict10), just output layer (conflict11), linear (conflict12), avg emb of neighboring words and with just one linear layer (conflict20, training of this gen is conflict18) run the experimens to collect/record result.

conflict14: record the performance of full encoder trained on all text in original and union of blankout trainset on the test set with the gold rationales as selcted only

run_conflict15: load union and original full encoder and (with no generator or with loading defferent sentence level (in contrast to word level in run_conflict10)  predictor geneartor like rcnn of tao lei (conflict15), just output layer (conflict16), linear (conflict17), avg_neighbor (conflict19) run the experimens to collect/record result.

conflict18: geneartor of just_output_layer with input: average( word_embeddings(word_t-2,t-1,t,t+1,t+2)) neighboring info
in contrast to the the simple word embedding of that word only (just_output_layer) and jointly train with an encoder. The genrator modeule can also be loaded (without the joint encoder) with different encoder (like full enc) afterwards (conflict20 and run_conflict10)

conflict21: load IMDB movie dataset, set evaluation and loss function accordingly, to train full enc ori + blank out, and then records results of same experiemnts (full enc, different gen + full enc(ori or blank out). Run by run_conflict21. Here the score is [0.0, 1,0] unlike conflict22 where there score is (0/1). This file is similar to conflict8. 

conflict22: load IMDB movie dataset, set evaluation and loss function accordingly, to train full enc, and then records results of same experiemnts (full enc, different gen + full enc). Run by run_conflict22. Here the score is (0/1) unlike conflict21 where there score is [0.0, 1,0]. This file is cross entropy loss related not required eventually, rather conflict21 is required. (not required eventualy)

conflict23: load Rotten Tomatoes movie dataset, set evaluation and loss function accordingly, to train full enc, and then records results of same experiemnts (full enc, different gen + full enc). Run by run_conflict23. Here the score is (0/1). (not required eventualy)

conflict24: Can load full enc and rcnn gen part from (rcnn gen + rcnn enc initialized with full enc) on rt and write the performance in a file. (not required eventualy)


run_rcnn_gen_rt: I can train jointly rcnn gen + rcnn enc, and later load only the rcnn gen part with the full enc from the other model by running rcnn_gen_rt.py on rt dataset. 

run_just_output_layer_rt: I can train jointly just output layer gen + rcnn enc (initialized with full enc), and later load only the rcnn gen part with the full enc from the other model from the output of this rcnn_gen.py

run_conflict_25: load union and original full rt encoder (conflict23) and (with no generator or with loading defferent geneartor like rcnn of tao lei_rt (rcnn_gen_rt.py), just output layerjust_output_layer_rt.py), run the experimens to collect/record result. similar to run_conflict10.

run_rcnn_gen_imdb: I can train jointly rcnn gen + rcnn enc (initialized with full enc ori+union), and later load only the rcnn gen part with the full enc from the other model from the output of this rcnn_gen.py

run_just_output_layer_imdb: I can train jointly just output layer gen + rcnn enc (initialized with full enc ori+union), and later load only the rcnn gen part with the full enc from the other model from the output of this rcnn_gen.py

run_conflict26: load union and original full imdb encoder and (with no generator or with loading defferent sentence level (in contrast to word level in run_conflict28)  predictor geneartor like rcnn of tao lei (conflict26), just output layer (conflict27), run the experimens to collect/record result. this file similar to run_conflict15.

run_conflict_28: load union and original full imdb encoder (conflict23) and (with no generator or with loading defferent geneartor like rcnn of tao lei_rt (rcnn_gen_imdb.py), just output layerjust_output_layer_imdb.py) unlike run_conflict26, run the experimens to collect/record result. similar to run_conflict10 and run_conflict25.
