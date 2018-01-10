
import os

lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]

# lamda_1 = [ 0.00025, 0.0003, 0.00035, 0.0004 ]

# lamda_1 = [ 0.0004]
lamda_2 = [2, 1]




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
graph_data_file = '../graph_data/full_enc_ori_just_output_layer.txt'
open(graph_data_file, 'w')


types = ['RCNN_RCNN','JUST_OUTPUT_LAYER']


_type = '../UNION_GEN/'
f = 0

base = '../../Budgeted_attention_model/rcnn/code/rationale/JUST_OUTPUT_LAYER'

gen_lr = {'JUST_OUTPUT_LAYER': 0.005, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.0005, 'AVG_LINEAR':0.005}

union = 'union_'
union = ''

for t in types:

	graph_data_file = '../graph_data/full_enc_'+union+t+'.txt'
	open(graph_data_file, 'w')

	for l_1 in lamda_1:
		for l_2 in lamda_2:

			path0 = base+'/Sampling/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'

			
			if(t=="AVG_LINEAR"): path = '../JUST_OUTPUT_LAYER/NEIGHBOR/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'

			if(t=="RCNN_RCNN"): path = '../RCNN_RCNN/Trained_MODELS/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr[t])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			for dp in dropout:
				
				for lr in [0.005]:#[ 0.0005, 0.005, 0.0001]:
					for select_all in [-1]:#[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

						model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						model_file_enc = '../RCNN_RCNN/CONFLICT8/model_sparsity_'+str(0)+'_coherent_'+str(0)+'_dropout_'+str(0.2)+"_lr_"+str(0.008)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
						

						py_file = 'conflict29.py'
						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu2,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +'  --embedding ../word_vec.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
						' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						+ ' --learning_rate '+str(lr)  +' --load_model ' + model_file_enc\
						 + ' --load_gen_model '+path0 + ' --save_model '	+ _type +'MODELS/'+union+model_file	 \
						 +' --sparsity '+ str(l_1) + ' --coherent '+str(l_2)		
						# run_command+= ' >> '+_type+union+ model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						exit()
	# exit()









# import os

# lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]
# # lamda_1 = [0.0002, 0.00025, 0.0003, 0.00035, 0.0004]

# # lamda_1 = [0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004] # for small 0.000085, 2 gru, markov, 
# # lamda_1 = [0.0003]#, 0.00006, 0.00005, 0.000065, 000075]

# # lamda_1 = [ 0.000085,  0.000105, 0.00011, 0.000085]
# # lamda_1 = [ 0.00035]
# # lamda_2 = [  0.8, 1, 1.5, 0.5, 0.1]
# lamda_2 = [  1, 2]
# # lamda_2 = [0.1]
# dropout = [0.1]
# dp = 0.1
# trained_max_epochs = 0
# load_emb_only = 1
# max_epochs = 100
# # learning = 'sgd'

# aspect = 1
# debug = 0.001
# select_all = -1
# output_file = 'rough2_new_gen_outputs_'+"linear_rcnn_0.01"+'.json'
# covered_percentage = []
# graph_data_file = '../graph_data/data_vs_lamda_table_dummy.txt'
# open(graph_data_file, 'w')
# base = '../../Budgeted_attention_model/rcnn/code/rationale/'

# _type = '../UNION_GEN/'
# gen_lr = {'JUST_OUTPUT_LAYER': 0.005, 'LINEAR_RCNN': 0.0005 , 'RCNN_RCNN': 0.0005, 'AVG_LINEAR':0.005}



# f = 0
# for l_1 in lamda_1:
# 	# f+=1
# 	# if(f>4):exit()
# 	for l_2 in lamda_2:
# 		for lr in [ 0.005]:


# 			path0 = '../RCNN_RCNN/Trained_MODELS/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr['RCNN_RCNN'])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
# 			path1 = base+'JUST_OUTPUT_LAYER'+'/Sampling/MODELS/model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(0.1)+"_lr_"+str(gen_lr['JUST_OUTPUT_LAYER'])+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			

# 			model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
# 			run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python conflict29.py --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --train ../reviews.aspect1.train.txt.gz --dev ../reviews.aspect1.heldout.txt.gz --embedding ../word_vec.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
# 				' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)+' --save_model ' +_type+ "MODELS/"+model_file \
# 				+' --load_model ' + path0 +' --load_gen_model ' + path1
			
# 			#run_command = 'python generator_fix.py --embedding word_vec.gz --load_rationale annotations.json --dump '+output_file+' --select_all ' +str(select_all)+ ' --aspect ' +str(aspect) +' --sparsity '+str(l_1)+' --coherent '+str(l_2)+' --load_model ' + 'model_new_generators/'+model_file #+ ' --graph_data_path '+ graph_data_file
			
# 			# run_command+= ' >> '+ _type+model_file + '.txt' 	
# 			print run_command
# 			os.system(run_command)
# 			print '\n\n\n'
# # 			exit()



# ### [0.0002, 0.0003, 0.0004] nlp 

# '''
# gru (my: 1, said avail: 1)
# ################
# l_1 = 0.000085, and 50% (before epoch 23 & l2 = 1)
# l_2 = 1, ( 2 cpu now)


# ######****###### l2 = 2 done (1 now)



# 2 gpu now 
# 1 cpu

# l_1 = 0.00035,
# l_2 = 1 cpu 


# ################
# margo (my: 3, others: 0)
# ################
# l_1 = 0.000105,
# l_2 = 1 (done), (2 done)

# l_1 = 0.0001,
# l_2 = 1 (done) (2 done)

# l_1 = 0.000095,
# l_2 = 2 (1 not yet)

# l_1 = 0.000095,
# l_2 = 1 (43) (2 on cpu almost done)


# l_1 = 0.00035,
# l_2 = 2 cpu 

# l_1 = 0.0004,
# l_2 = 1 (3), 


# l_1 = 0.00025,
# l_2 = 1 (5), 




# ######****###### 

# l_1 = 0.000105,
# l_2 = 1 (done), (done 2)

# l_1 = 0.0001,
# l_2 = 1 (now) (2 done)
 
# l_1 = 0.000095,
# l_2 = 2 now (1 now)





# ################

# nlp (my: 3, others: 0)
# ################
# l_1 = 0.00012, 
# l_2 = 1 (42) (2 done) 

# l_1 = 0.000115, 
# l_2 = 1 (43) cpu

# l_1 = 0.000115, 
# l_2 = 2 (30) cpu now 


# l_1 = 0.00011, 
# l_2 = 1 (43) (2 done)  

# ################

# ######****######
# l_1 = 0.00012, 
# l_2 = 1 (done) (2 done) 

# l_1 = 0.000115, 
# l_2 = 1 (now) 2 (done)

# l_1 = 0.00011, 
# l_2 = 1 (now) (2 now)  



# svm (my: 2, others: 1)
# ################
# l_1 = 0.00016, 
# l_2 = 1 (42) (2 done) 

# l_1 = 0.0002,
# l_2 = 1 (42) (2 done)

# l_1 = 0.00025,
# l_2 = 1 (5 now on margo) , 2 (27) now cpu  (2, 1 not yet)


# ######****######
# l_1 = 0.00016, 
# l_2 = 1 (done) (2 done) 

# l_1 = 0.0002,
# l_2 = 1 (now) (2 now)

# l_1 = 0.00025, (both 1, 2 NOW ON POWER$ NEW GEN)

# ################

# crf (my: 2, others: 0) [totl gpu = 2]
# ################
# l_1 = 0.0004,
# l_2 = 1 (3 on margo gpu), 2 (21) now cpu (1, 2 not yet)

# l_1 = 0.00035,
# l_2 = now cpu  (2, 1 not yet)

# 1 in margo cpu
# 2 in gru cpu

# l_1 = 0.0003, (NOW ON POWER$ NEW GEN)
# l_2 = 1 (21) (2 done)


# ################
# l_1 = 0.0004,
# l_2 = 1 (done), 2 (done)

# l_1 = 0.00035,
# l_2 = 2 (done)  1 (done)


# l_1 = 0.0003, (NOW ON POWER$ NEW GEN)

# l_1 = 0.00025,
# l_2 = 1 (now) (2 now)

# power4:
# ################

# l_1 = 0.0003,
# l_2 = 1 (now) (2 done)

# '''

