import os

lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]

# lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0005 ]

# lamda_1 = [ 0.000085, 0.000095]
lamda_2 = [2, 1]




dropout = [ 0.1]#, 0.2]
# dp = 0.1
trained_max_epochs = 0

max_epochs = 100
# learning = 'sgd'
aspect = 1
debug = 1
select_all = -1
output_file = 'rough2_new_gen_outputs_'+"linear_rcnn_0.01"+'.json'
covered_percentage = []
graph_data_file = '../graph_data/full_enc_ori_just_output_layer.txt'
open(graph_data_file, 'w')

types = ['JUST_OUTPUT_LAYER']

_type = '../JUST_OUTPUT_LAYER/CONCATE_NEIGHBOR/'
f = 0

base = '../../Budgeted_attention_model/rcnn/code/rationale/'
gen_lr = {'JUST_OUTPUT_LAYER': 0.005, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.0005}



for t in types:

	for l_1 in lamda_1:
		for l_2 in lamda_2:

			for dp in dropout:
				
				for lr in [0.005]:#[ 0.0005, 0.005, 0.0001]:#to train gen with lr of 'JUST_OUTPUT_LAYER': 0.005
					for select_all in [-1]:#[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

						model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						
						
						py_file = 'conflict19.py'

						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --train ../reviews.aspect1.train.txt.gz --dev ../reviews.aspect1.heldout.txt.gz  --load_rationale ../annotations.json --aspect ' + str(aspect) + \
						' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						+ ' --learning_rate '+str(lr)  +' --save_model ' + _type +'MODELS/'+model_file #+ ' --debug 0.001' 
						
						# run_command+= ' >> '+_type+model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						exit()
	# exit()


