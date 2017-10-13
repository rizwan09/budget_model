import os

#lamda_1 = [0.0002]
# lamda_1 = [0.00012]
#lamda_1 = [0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0005, 0.00001, 0.00005, 0.00009, 00012, 0.00016, 0.000005]

lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]# 0.000085, 0.000095, 0.0001, 0.00016, 0.00025, 0.0004
# lamda_1 = [0.0004]#, 0.0003, 0.00035, 0.0002, 0.00025, 0.00016, 0.00012, 0.000115, 0.00011, 0.000105, 0.0001]# 0.000085, 0.000095, 0.0001, 0.00016, 0.00025, 0.0004
# lamda_1 = [0.00007, 0.00006, 0.00005, 0.000065, 000075]
# lamda_2 = [  0.8, 2, 1.5, 1, 0.1, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.9, 0.95]
lamda_2 = [  0.1, 0.25, 0.35, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95,  1, 1.5, 2]

dp = 0.1
learning_rate = [0.0005]
max_epochs = 100
aspect = 1
select_all = -1
output_file = 'outputs_2.json'
graph_data_file = 'graph_data/'
#open(graph_data_file, 'w')
# for ext in range(0, 100, 5):
# lamda_1 = [0.00025]
# lamda_2 = [1]
# select_all_array = [0.95, 0.96, 0.945, 0.955, 0.965, 0.948, 0.958, 0.962, 0.953 ]
select_all_array = [-1]
for l_1 in lamda_1:
	for l_2 in lamda_2:
		# if(l_1 in [0.00025, 0.0003, 0.00035 ]and l_2 in [0.65, 0.6, 0.7, 0.75]): continue
		for lr in learning_rate:
			for path in ['JUST_OUTPUT_LAYER']:#, 'RCNN_RCNN', 'LINEAR_RCNN' ]:
				graph_data_file = '../graph_data/'+"tabel_"+path+"_"+"FIX_ENC_TRAIN_RCNN_GEN2_with_recall.txt"#+str(select_all)+".txt"
				model_file = "../"+path+"/FIX_ENC_TRAIN_RCNN_GEN2/MODELS/"+'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
				# if(ext > 0): model_file+= str(ext)
				load_model_file = model_file#'models/'+model_file #+str(trained_max_epochs-1)+'.pkl.gz'
				#model_file = 'models/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)#+'_epochs_'+str(trained_max_epochs) #+'.txt.pkl.gz'
				#load_model_file = 'models/'+model_file  +'_epochs_'+str(trained_max_epochs)+'.txt.pkl.gz'
				#model_file = model_file+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'+str(cur_epoch)+'.pkl.gz'
				#model_file = 'model_sparsity_'+str(l_1)+'_epochs_'+str(max_epochs)+'.txt.pkl.gz'
				#model_file = 'models/'+model_file
				if not os.path.exists(load_model_file): 
					# continue
					print 'not exist: ', model_file
					continue
				if(path == 'RCNN_RCNN'): py_file = 'gen_enc'
				else: py_file= path.lower()
				# run_command = 'THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+py_file+'_RSGEN.py --embedding ../word_vec.gz --load_rationale ../annotations.json --dump ../outputs_with_first_loading.json --select_all ' +str(select_all)+ ' --aspect ' +str(aspect) +' --sparsity '+str(l_1)+' --coherent '+str(l_2)+' --load_model ' + load_model_file  + ' --graph_data_path '+ graph_data_file + ' --learning_rate '+ str(lr) 
				for select_all in select_all_array :
					run_command = 'THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python fix_enc_train_rcnn_gen_with_recall.py --embedding ../word_vec.gz --load_rationale ../annotations.json --dump ../outputs_with_first_loading.json --select_all ' +str(select_all)+ ' --aspect ' +str(aspect) +' --sparsity '+str(l_1)+' --coherent '+str(l_2)+' --load_model ' + load_model_file  + ' --graph_data_path '+ graph_data_file + ' --learning_rate '+ str(lr) 
					print run_command
					# run_command+=' >> '+ 'graph_data/'+ 'NOT_SELECTED_RACTIONALS'+'.txt' 
					os.system(run_command)
					print '\n\n\n '
				# exit()

#now on nlp 13807