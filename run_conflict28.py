import os

lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]

# lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0005 ]

# lamda_1 = [ 0.0003]
lamda_2 = [1, 2]




dropout = [ 0.2]
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

types = ['RCNN_RCNN','JUST_OUTPUT_LAYER']

_type = '../IMDB/'
f = 0

base = '../../Budgeted_attention_model/rcnn/code/rationale/'
gen_lr = {'JUST_OUTPUT_LAYER': 0.001, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.001, 'AVG_LINEAR': 0.005}

union = 'union_'
union = ''
num_data = 5
d = d2 =128
batch_size  =50
l2_reg  =1e-6
l='rcnn'
for t in types:
	
	graph_data_file = '../graph_data/imdb_full_enc_n_word_gen_'+union+t+'.txt'
	open(graph_data_file, 'w')

	for l_1 in lamda_1:
		for l_2 in lamda_2:

			for dp in dropout:
				
				for lr in [0.001]:#[ 0.0005, 0.005, 0.0001]:
					for select_all in [-1]:#[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
						if(t=="RCNN_RCNN"): 
							dp = 0.2
							path = _type+'RCNN_RCNN/MODELS/'+union+'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						if(t=="JUST_OUTPUT_LAYER"): 
							dp = 0.2
							path = _type+'JUST_OUTPUT_LAYER/MODELS/'+union+'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			
						if union!='':
							assert lr == 0.001 ## change this
							assert dp == 0.2 ## change this
							assert d2 ==128 ## change this
							assert d == 128 ## change this
							assert batch_size==50 ## change this
							# assert l2_reg == 1e-6

						load_model_file = 'model_'+l+'_sparsity_0_coherent_0_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_l2_'+str(l2_reg)+ '_batch_'+str(batch_size)+'_d_'+str(d)+'_d2_'+str(d2)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						# model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			
						# else:
						# 	assert load_model_file != ''
						if t=='JUST_OUTPUT_LAYER':
							py_file = 'just_output_layer_imdb.py'
						elif t=='RCNN_RCNN':
							py_file = 'rcnn_gen_imdb.py'
						

						# run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
						# ' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						# + ' --learning_rate '+str(lr)  +' --load_model ' + _type +'MODELS/'+union+load_model_file+' --num_data '+str(num_data) \
						#  + ' --load_gen_model '+path +' --graph_data_path '+ graph_data_file 
						

						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+ py_file  +' --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --batch 50 -d 128 -d2 128 --layer '+l+' --test imdb  --embedding glove.6B.300d_w_header.txt' + \
						' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)\
						+ ' --load_model ' + _type +'MODELS/'+union+load_model_file + ' --load_gen_model ' + path
			


						# run_command+= ' >> '+_type+union+ model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						exit()
	# exit()



