import os

lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]

# lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0005 ]

# lamda_1 = [ 0.00016]
lamda_2 = [2,1]




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
graph_data_file = '../graph_data/rough.txt'
open(graph_data_file, 'w')


types = ['RCNN_RCNN','JUST_OUTPUT_LAYER','LINEAR_RCNN', 'AVG_LINEAR']
types = ['JUST_OUTPUT_LAYER','RCNN_RCNN']

_type = '../RCNN_RCNN/CONFLICT8/'
f = 0

base = '../../Budgeted_attention_model/rcnn/code/rationale/'

gen_lr = {'JUST_OUTPUT_LAYER': 0.005, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.0005, 'AVG_LINEAR':0.005}

union = 'union_'
# union = ''
num_data = 5

for t in types:

	
	if(t!="JUST_OUTPUT_LAYER"):
		continue
	graph_data_file = '../graph_data/full_enc_ori_'+union+t+'.txt'
	open(graph_data_file, 'w')

	for l_1 in lamda_1:
		for l_2 in lamda_2:

			path = base+t+'/Sampling/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'

			
			if(t=="AVG_LINEAR"): path = '../JUST_OUTPUT_LAYER/NEIGHBOR/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'

			if(t=="RCNN_RCNN"): path = '../RCNN_RCNN/Trained_MODELS/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			for dp in dropout:
				
				for lr in [0.008]:#[ 0.0005, 0.005, 0.0001]:
					for select_all in [-1]:#[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

						if union!='':
							dp = 0.3
							lr = 0.0001
							assert num_data>0

						model_file = 'model_sparsity_'+str(0)+'_coherent_'+str(0)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						# load_model_file = _type+'MODELS/'+model_file + str(trained_max_epochs)
						load_model_file  =''
						if num_data>0:
							assert union !=''
							assert trained_max_epochs !=95
							assert load_model_file == ''
							assert dp == 0.3
							assert lr == 0.0001
						# else:
						# 	assert load_model_file != ''
						if t=='LINEAR_RCNN':
							py_file = 'conflict12.py'
						elif t=='JUST_OUTPUT_LAYER':
							py_file = 'conflict11.py'
						elif t=='RCNN_RCNN':
							py_file = 'conflict10.py'

						elif t=='AVG_LINEAR':
							py_file = 'conflict20.py'

						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
						' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						+ ' --learning_rate '+str(lr)  +' --load_model ' + _type +'MODELS/'+union+model_file+' --num_data '+str(num_data) \
						 + ' --load_gen_model '+path +' --graph_data_path '+ graph_data_file +' --sparsity '+str(l_1) +' --coherent '+str(l_2)
						
						# run_command+= ' >> '+_type+union+ model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						# exit()
	# exit()



