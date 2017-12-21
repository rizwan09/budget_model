import os

# lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]


lamda_1 = [ 0.0003]
lamda_2 = [2]




dropout =[ 0.3]#, 0.05, 0.2]
# dp = 0.1
trained_max_epochs = 00
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

types = ['RCNN_RCNN']#,'JUST_OUTPUT_LAYER']#,'LINEAR_RCNN','AVG_LINEAR']

_type = '../IMDB/'
f = 0

union = 'union_'
union = ''
num_data = 0

for d in [128]:
	t = types[0]
	for dp in dropout:
		l_1 = 0
		l_2 = 0
		for lr in [0.005]:#0.001: #in [0.01]:#, 0.05]:#, 0.0009]:
			batch_size  = 50
			
			# load_model_file = _type+'MODELS/'+model_file + str(trained_max_epochs)
			load_model_file  = ''
			if num_data>0:
				assert union !=''
				assert trained_max_epochs !=95
				assert load_model_file == ''
			# else:
			# 	assert load_model_file != ''
			d2 = 128
			for l2_reg in [1e-6]:
			#l in [ 'lstm']:#,'lstm]:
				l = 'rcnn'
				# if dp==0.1 and l=='lstm':continue
				model_file = 'model_'+l+'_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_l2_'+str(l2_reg)+ '_batch_'+str(batch_size)+'_d_'+str(d)+'_d2_'+str(d2)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
				# model_file = 'model_'+l+'_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
				
				run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" python conflict21.py --layer '+l+' -d '+str(d)+' -d2 128 --sparsity 0 --coherent 0 --train imdb --dev imdb --test imdb --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --embedding glove.6B.300d_w_header.txt' +\
					' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --batch '+str(batch_size)+' --learning_rate '+str(lr)  +' --save_model ' + _type+'/MODELS/'+union+model_file+' --num_data '+str(num_data)+' --l2_reg '+str(l2_reg) # + ' --load_model '+_type+'MODELS/'+union+model_file
				
				# run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu2,floatX=float32" python conflict21.py --layer '+l+' -d2 128 --sparsity 0 --coherent 0 --test imdb --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --embedding glove.6B.300d_w_header.txt' +\
				# 	' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all)+' --learning_rate '+str(lr)  +' --load_model ' + _type+'MODELS_OLD/'+union+model_file # +' --num_data '+str(num_data)+' --l2_reg '+str(l2_reg)  + ' --load_model '+_type+'MODELS/'+union+model_file
				

				run_command+= ' >> '+_type+'MODELS/'+union+ model_file +'.txt'
				print run_command 
				os.system(run_command)
				print '\n\n\n'
				# exit()




