import os


lamda_1 = [-3, -2 , -1, 0.001, 0.0001, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009,0.008,0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0001, 0.0009]
# lamda_1 = [0.001, 0.0001, -1, -2 , -3]
lamda_2 = [ 1, 0.5, -2 , 0, -1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1, 0.5, ]
# lamda_2 = [ 1, 0.5, -2 , 0, -1]



dropout = [ 0.1]
# dp = 0.1
trained_max_epochs = 0

max_epochs = 100
# learning = 'sgd'
aspect = 1
debug = 1
select_all = -1
output_file = 'rough2_new_gen_outputs_'+"linear_rcnn_0.01"+'.json'
covered_percentage = []
# graph_data_file = '../graph_data/rt_full_enc_ori_rcnn_gen.txt'
# open(graph_data_file, 'w')

types = ['JUST_OUTPUT_LAYER','RCNN_RCNN']

_type = '../ROTTEN_TOMATOES/'
f = 0

base = '../../Budgeted_attention_model/rcnn/code/rationale/'
gen_lr = {'JUST_OUTPUT_LAYER': 0.005, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.0005, 'AVG_LINEAR':0.005}

l='rcnn'
l='lstm'

l2_reg = 1e-6
batch_size = 50
# d = 200
# d2= 128
# union = 'union_'
union = ''
num_data = 0

for t in types:

	graph_data_file = '../graph_data/rt_ori_'+l+'_enc_'+union+t+'_gen_new.txt'
	open(graph_data_file, 'w')

	for l_1 in lamda_1:
		for l_2 in lamda_2:

			for dp in dropout:
				
				for lr in [0.0005]:#[ 0.0005, 0.005, 0.0001]:
					for select_all in [-1]:#[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

						if union!='':
							dp = 0.3
							lr = 0.0001
							assert num_data>0

						load_model_file = 'model_'+l+'_sparsity_0_coherent_0_dropout_'+str(0.05)+"_lr_"+str(0.0009)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						# load_model_file = _type+'MODELS/'+model_file + str(trained_max_epochs)
						if os.path.exists(_type+t+'/MODELS/'+model_file)==False: continue
						# load_model_file = 'model_'+l+'_sparsity_0_coherent_0_dropout_'+str(0.1)+"_lr_"+str(0.0005)+'_full_trainset_l2_'+str(l2_reg)+ '_batch_'+str(batch_size)+'_d_'+str(d)+'_d2_'+str(d2)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						enc_model = _type+'MODELS_old/'+load_model_file
						if l=='lstm': enc_model = _type+'MODELS/'+load_model_file #both gen enc from same file
						if num_data>0:
							assert union !=''
							assert trained_max_epochs !=95
							assert load_model_file == ''
							assert dp == 0.3
							assert lr == 0.0001
						# else:
						# 	assert load_model_file != ''
						
						if t == 'JUST_OUTPUT_LAYER' :
							py_file = 'just_output_layer_rt.py'
						elif t=='RCNN_RCNN':
							py_file = 'rcnn_gen_rt.py'
						

						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu2,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +' --test rotten_tomatoes  --embedding glove.6B.300d_w_header.txt'+ \
						' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						+ ' --learning_rate '+str(lr) +' --load_model ' +enc_model+' --num_data '+str(num_data) +' --gen_type '+t\
						 + ' --load_gen_model '+_type+t+'/MODELS/'+model_file+' --graph_data_path '+ graph_data_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+str(dp)\
						 +' --layer '+l
						
						# run_command+= ' >> '+_type+union+ model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						# exit()

	# exit()



