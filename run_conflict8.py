import os

# lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]


lamda_1 = [ 0.0003]
lamda_2 = [2]




dropout = [ 0.2, 0.3]
# dp = 0.1
trained_max_epochs = 0
load_emb_only = 1
max_epochs = 100
# learning = 'sgd'
aspect = 1
debug = 1
select_all = 1
output_file = 'rough2_new_gen_outputs_'+"linear_rcnn_0.01"+'.json'
covered_percentage = []
graph_data_file = '../graph_data/conflict6_annotations0.1to0.9.txt'
open(graph_data_file, 'w')

types = ['JUST_OUTPUT_LAYER', 'LINEAR_RCNN', 'RCNN_RCNN', 'just_output_layer', 'linear_rcnn']

_type = '../RCNN_RCNN/CONFLICT8/'
f = 0

union = 'union_'
union = ''
num_data = 0

for dp in dropout:
		l_1 = 0
		l_2 = 0
		for lr in [0.01, 0.009]:#[ 0.0005, 0.005, 0.0001]:
			
			model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			# load_model_file = _type+'MODELS/'+model_file + str(trained_max_epochs)
			load_model_file  =''
			if num_data>0:
				assert union !=''
				assert trained_max_epochs !=95
				assert load_model_file == ''
			# else:
			# 	assert load_model_file != ''


			run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python conflict8.py --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --train ../reviews.aspect1.train.txt.gz --dev ../reviews.aspect1.heldout.txt.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
				' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)  +' --save_model ' + _type +'MODELS/'+union+model_file+' --num_data '+str(num_data) #+ ' --load_model '+load_model_file
			
			run_command+= ' >> '+_type+union+ model_file +'.txt'
			print run_command 
			os.system(run_command)
			print '\n\n\n'
			# exit()




