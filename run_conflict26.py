import os


def get_selection_beforehand(file):
	# print file
	with open (file, 'r') as f:
		l1 = -1
		l2=-1
		is_best = 0
		s =-1
		e=-1
		g=-1
		tt=-1
		ac=-1
		fg = 0
		fe = 0
		dev_found = 0
		dev_s_found = 0
		c =0
		for line in f:

			if 'saving best model' in line:
				is_best = 1 #make 0 again
				c = c+1
				# print line


			for word in line.split():
				if 'sparsity=' in word:
					l1= word.split('sparsity=')[1].strip(',')
				if 'coherent=' in word:
					l2= word.split('coherent=')[1].strip(',')
				if 'accuracy=' in word and is_best==1:
					# print line
					if dev_found==0:
						dev_found=1
						# print 'dev found'
					else:
						# dev_found=0
						ac = word.split('accuracy=')[1].strip(',')
						
				if 'p[1]g=' in word and dev_found==1:
					s = word.split('p[1]g=')[1].strip(',')

				if 'time=' in word and dev_found==1 and fg==1 and fe==0:
					e = word.split('time=')[1].strip(',')
					fe=1
					continue
				if 'time=' in word and dev_found==1 and fg==0 and fe==0:
					g = word.split('time=')[1].strip(',')
					fg=1
					continue
				if 'time=' in word and dev_found==1 and fe==1 and fg==1:
					# print word, "\n IN: ", line
				
					dev_found = 0
					tt = word.split('time=')[1].strip(',')
					is_best = 0
					fg = 0
					fe = 0
	print file, ' returning s: ', s	
	return float(s)				



lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075, 0.0008, 0.00085, 0.0009, 0.001, 0.0001, -1, -2, -3, -4]
lamda_1 = lamda_1[::-1]



lamda_2 = [-4, -5, -3, -2, -1, -.5, -0.1, 0, 0.25, 0.5, 0.75, 0.85, 0.95, -0.25, -0.75, -0.85, -0.95, -2.75, 1, 2]#[  0.5]





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
# graph_data_file = '../graph_data/full_enc_ori_just_output_layer.txt'
# open(graph_data_file, 'w')

types = ['JUST_OUTPUT_LAYER', 'RCNN']

_type = '../IMDB/'
f = 0

base = '../../Budgeted_attention_model/rcnn/code/rationale/'
gen_lr = {'JUST_OUTPUT_LAYER': 0.001, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.001, 'AVG_LINEAR': 0.005}

union = 'union_'
# union = ''
# num_data = 5
d = 200
d2 =30
batch_size  =256
l2_reg  =1e-6
l='rcnn'
for t in types:
	graph_data_file = '../graph_data/2Sen_dummy_'+union+t+'.txt'
	open(graph_data_file, 'w')

	for l_1 in lamda_1:
		for l_2 in lamda_2:
			# if t!='RCNN_RCNN' or l_1!=-3 or l_2!=0.85:continue
	
			for dp in dropout:
				
				for lr in [0.001]:#[ 0.0005, 0.005, 0.0001]:
					for select_all in [1]:#[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
						if(t=="RCNN_RCNN"): 
							dp = 0.1
							path = _type+'RCNN_RCNN/MODELS/'+union+'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						if(t=="JUST_OUTPUT_LAYER"): 
							dp = 0.1
							path = _type+'JUST_OUTPUT_LAYER/MODELS/'+union+'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
							# print ' in jol'
						# if union!='':
						# 	assert lr == 0.001 ## change this
						# 	assert dp == 0.1 ## change this
							# assert d2 ==128 ## change this
							# assert d == 128 ## change this
							# assert batch_size==50 ## change this
							# assert l2_reg == 1e-6

						load_model_file = 'model_'+l+'_sparsity_0_coherent_0_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						# model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			
						# else:
						# 	assert load_model_file != ''
						if t=='JUST_OUTPUT_LAYER':
							py_file = 'conflict27.py'
						elif t=='RCNN_RCNN':
							py_file = 'conflict26.py'
						

						# run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
						# ' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						# + ' --learning_rate '+str(lr)  +' --load_model ' + _type +'MODELS/'+union+load_model_file  \
						#  + ' --load_gen_model '+path +' --graph_data_path '+ graph_data_file + ' --sparsity '+ str(l_1) + ' --coherent ' + str(l_2)\
						#  + ' --dropout '+ str(dp) + ' --learning_rate '+str(lr)
						print t, path, os.path.exists(path)
						if os.path.exists(path)==False: continue
						print'ok got it'
						s = get_selection_beforehand(path.replace('/MODELS', '')+'.txt')
						# print'ok got it'
						if s<0.05 or s>0.90: continue
						# print'ok got it'
						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" python '+ py_file  +' --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --batch '+str(batch_size)+' -d '+str(d)+' -d2 '+str(d2)+' --layer '+l+' --test imdb  --embedding glove.6B.300d_w_header.txt' + \
						' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)\
						+ ' --load_model ' + _type +'MODELS/'+union+load_model_file + ' --load_gen_model '+path +' --graph_data_path '+ graph_data_file + ' --sparsity '+ str(l_1) + ' --coherent ' + str(l_2)\
						 + ' --dropout '+ str(dp) + ' --learning_rate '+str(lr)
			


						# run_command+= ' >> '+_type+union+ model_file +'.txt'
						print run_command, "\n", t
						os.system(run_command)
						print '\n\n\n'
						# exit() 
	exit()



