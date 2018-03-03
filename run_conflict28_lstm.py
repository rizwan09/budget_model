import os

lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075, 0.0008, 0.00085, 0.0009, 0.001, 0.0001, -1, -2, -3, -4]
lamda_1 = lamda_1[::-1]



lamda_2 = [-4, -5, -3, -2, -1, -.5, -0.1, 0, 0.25, 0.5, 0.75, 0.85, 0.95, -0.25, -0.75, -0.85, -0.95, -2.75, -2, -1, 1, 2]#[  0.5]





dropout = [ 0.1]
# dp = 0.1
trained_max_epochs = 0

max_epochs = 100
# learning = 'sgd'
aspect = 1
debug = 1
select_all = 1
output_file = 'rough2_new_gen_outputs_'+"linear_rcnn_0.01"+'.json'
covered_percentage = []
# graph_data_file = '../graph_data/full_enc_ori_just_output_layer.txt'
# open(graph_data_file, 'w')

types = ['JUST_OUTPUT_LAYER', 'RCNN_RCNN']

_type = '../IMDB/'
f = 0

base = '../../Budgeted_attention_model/rcnn/code/rationale/'
gen_lr = {'JUST_OUTPUT_LAYER': 0.001, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.001, 'AVG_LINEAR': 0.005}

union = 'union_'
# union = ''
num_data = 5
d = 128
# d2 =30
batch_size  = 128
l2_reg  =1e-6
# l='rcnn'
l='lstm'
for t in types:
	graph_data_file = '../graph_data/dummy_imdb_'+t+'_gen_'+union+''+l+'_enc.txt'
	open(graph_data_file, 'w')

	for l_1 in lamda_1:
		for l_2 in lamda_2:
			# if t!='RCNN_RCNN' or l_1!=-3 or l_2!=0.85:continue
	
			for dp in dropout:
				
				for lr in [0.008]:#[ 0.0005, 0.005, 0.0001]:
					for select_all in [-1]:#[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
						if(t=="RCNN_RCNN"): 
							dp = 0.1
							path = _type+'RCNN_RCNN_old/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
							# path = _type+'RCNN_RCNN_old/MODELS/'+union+'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						if(t=="JUST_OUTPUT_LAYER"): 
							dp = 0.1
							path = _type+'JUST_OUTPUT_LAYER_old/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			
						load_model_file = 'model_'+l+'_sparsity_'+str(0)+'_coherent_'+str(0)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'_batch_'+str(batch_size)+'_d_'+str(d)+'.txt.pkl.gz'#+'_depth_'+str(depth)+'.txt.pkl.gz'
						if union!='':
							assert lr == 0.008 ## change this
							# assert dp == 0.1 
							load_model_file = 'model_'+l+'_sparsity_'+str(0)+'_coherent_'+str(0)+'_dropout_'+str(0.2)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'_batch_'+str(batch_size)+'_d_'+str(d)+'.txt.pkl.gz'#+'_depth_'+str(depth)+'.txt.pkl.gz'
						

						# model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			
						# else:
						# 	assert load_model_file != ''
						if t=='JUST_OUTPUT_LAYER':
							py_file = 'just_output_layer_lstm_enc_imdb.py'
						elif t=='RCNN_RCNN':
							py_file = 'lstm_gen_lstm_enc_imdb.py'
						

						# run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
						# ' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						# + ' --learning_rate '+str(lr)  +' --load_model ' + _type +'MODELS/'+union+load_model_file  \
						#  + ' --load_gen_model '+path +' --graph_data_path '+ graph_data_file + ' --sparsity '+ str(l_1) + ' --coherent ' + str(l_2)\
						#  + ' --dropout '+ str(dp) + ' --learning_rate '+str(lr)
						if os.path.exists(path)==False: continue
						

						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" python '+ py_file  +' --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --batch '+str(batch_size)+' -d '+str(d)+' --layer '+l+' --test imdb  --embedding glove.6B.300d_w_header.txt' + \
						' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)\
						+ ' --load_model ' + _type +'LSTM/MODELS/'+union+load_model_file + ' --load_gen_model '+path +' --graph_data_path '+ graph_data_file + ' --sparsity '+ str(l_1) + ' --coherent ' + str(l_2)\
						 + ' --dropout '+ str(dp) + ' --learning_rate '+str(lr)
			


						# run_command+= ' >> '+_type+union+ model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						exit()
	# exit()



