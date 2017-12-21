import os

# lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]
# lamda_1 = [ 0.00025, 0.0003, 0.00035, 0.0004]

# lamda_1 = [0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004] # for small 0.000085, 2 gru, markov, 
# lamda_1 = [0.0003]#, 0.00006, 0.00005, 0.000065, 000075]

# lamda_1 = [ 0.000085,  0.000105, 0.00011, 0.000085]
# lamda_1 = [ 0.00035]
# lamda_2 = [  0.8, 1, 1.5, 0.5, 0.1]
# lamda_2 = [  0.1, 0.25, 0.35, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95,  1, 1.5, 2]


lamda_1 = [ 0.0003, 0.0001, 0.00012]
lamda_2 = [0.8, 0.5, 1]




dropout = [0.1]
dp = 0.1
trained_max_epochs = 0
load_emb_only = 0
max_epochs = 100
# learning = 'sgd'
aspect = 1
debug = 1
select_all = -1
output_file = 'rough2_new_gen_outputs_'+"linear_rcnn_0.01"+'.json'
covered_percentage = []
graph_data_file = '../graph_data/data_vs_lamda_table_dummy.txt'
open(graph_data_file, 'w')

types = ['JUST_OUTPUT_LAYER', 'LINEAR_RCNN', 'RCNN_RCNN', 'just_output_layer', 'linear_rcnn']

_type = '../RCNN_RCNN/CONFLICT4/'
f = 0
# for l_1 in lamda_1:
	# f+=1
	# if(f>4):exit()
	# for l_2 in lamda_2:
for f in range(3):
		l_1 = lamda_1[f]
		l_2 = lamda_2[f]
		for lr in [ 0.0005]:
			# if(f==1):continue
			# if l_1 == 0.00035 or l_1==0.0003 :continue
			# if l_1 == 0.00035 and l_2 == 2: continue
			# if(l_1 in [0.00025, 0.0003, 0.00035 ]and l_2 in [0.65, 0.6, 0.7, 0.75]): continue
			model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			load_model_file = "../RCNN_RCNN/"+'MODELS/'+'model_sparsity_'+str(0.0003)+'_coherent_'+str(1)+'_dropout_'+str(0.1)+"_lr_"+str(0.005)+'_max_epochs_'+str(100)+'.txt.pkl.gz' #+str(trained_max_epochs-1)+'.pkl.gz'
			# model_file = 'removed_redundancy_'+
			# load_model_file = '../JUST_OUTPUT_LAYER/FIX_ENC_TRAIN_RCNN_GEN2/'+"MODELS/"+model_file
			#run_command = 'python generator_fix.py --max_epochs '+ str(max_epochs) + ' --embedding word_vec.gz   --load_rationale annotations.json --aspect ' + str(aspect) + \
			# 	' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp) +' --load_model '+ load_model_file  #+' --save_model model_new_generators/' + model_file # +' --load_model '+ load_model_file
			#run_command = 'THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" python generator_fix.py --embedding word_vec.gz --load_rationale annotations.json --dump outputs_with_first_loading.json --select_all ' +str(select_all)+ ' --aspect ' +str(aspect) +' --sparsity '+str(l_1)+' --coherent '+str(l_2)+' --load_model ' + load_model_file  + ' --graph_data_path '+ graph_data_file #+ ' --cur_epoch '+ str(cur_epoch)
			

			# run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python just_output_layer.py --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) + ' --embedding word_vec.gz --load_rationale annotations.json --aspect ' + str(aspect) + \
			# 	' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --load_model ' + load_model_file +' --select_all '+ str(select_all) + ' --load_emb_only '+ str(load_emb_only)
			
			# run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python gen_enc.py --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) + ' --embedding word_vec.gz --train reviews.aspect1.train.txt.gz --dev reviews.aspect1.heldout.txt.gz --load_rationale annotations.json --aspect ' + str(aspect) + \
			# 	' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)+' --save_model ' +_type+ "MODELS/"+model_file +' --load_model '+ load_model_file + ' --load_emb_only '+ str(load_emb_only)
			
			#rcnn_gen_fix_enc.py
			# RAGN = Reinforcment Adam generator
			# RSGN = Reinforcment SGD generator
			# run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python just_output_layer_fix_enc_train_rcnn_gen.py --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) + ' --embedding ../word_vec.gz --train ../reviews.aspect1.train.txt.gz --dev ../reviews.aspect1.heldout.txt.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
			# 	' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)+' --save_model ' +_type+ "MODELS/"+model_file +' --load_model '+ load_model_file + ' --load_emb_only '+ str(load_emb_only)
			
			run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu2,floatX=float32" python conflict4.py --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --train ../reviews.aspect1.train.txt.gz --dev ../reviews.aspect1.heldout.txt.gz --load_rationale ../annotations.json --aspect ' + str(aspect) + \
				' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr) +' --save_model ' +_type+ "MODELS/"+model_file+' --load_model '+ load_model_file  + ' --load_emb_only '+ str(load_emb_only)
			
			#run_command = 'python generator_fix.py --embedding word_vec.gz --load_rationale annotations.json --dump '+output_file+' --select_all ' +str(select_all)+ ' --aspect ' +str(aspect) +' --sparsity '+str(l_1)+' --coherent '+str(l_2)+' --load_model ' + 'model_new_generators/'+model_file #+ ' --graph_data_path '+ graph_data_file
			
			run_command+= ' >> '+_type+ model_file +'.txt' 	
			print run_command
			os.system(run_command)
			print '\n\n\n'
			# exit()



### [0.0002, 0.0003, 0.0004] nlp 

'''
gru (my: 1, said avail: 1)
################
l_1 = 0.000085, and 50% (before epoch 23 & l2 = 1)
l_2 = 1, ( 2 cpu now)


######****###### l2 = 2 done (1 now)



2 gpu now 
1 cpu

l_1 = 0.00035,
l_2 = 1 cpu 


################
margo (my: 3, others: 0)
################
l_1 = 0.000105,
l_2 = 1 (done), (2 done)

l_1 = 0.0001,
l_2 = 1 (done) (2 done)

l_1 = 0.000095,
l_2 = 2 (1 not yet)

l_1 = 0.000095,
l_2 = 1 (43) (2 on cpu almost done)


l_1 = 0.00035,
l_2 = 2 cpu 

l_1 = 0.0004,
l_2 = 1 (3), 


l_1 = 0.00025,
l_2 = 1 (5), 




######****###### 

l_1 = 0.000105,
l_2 = 1 (done), (done 2)

l_1 = 0.0001,
l_2 = 1 (now) (2 done)
 
l_1 = 0.000095,
l_2 = 2 now (1 now)





################

nlp (my: 3, others: 0)
################
l_1 = 0.00012, 
l_2 = 1 (42) (2 done) 

l_1 = 0.000115, 
l_2 = 1 (43) cpu

l_1 = 0.000115, 
l_2 = 2 (30) cpu now 


l_1 = 0.00011, 
l_2 = 1 (43) (2 done)  

################

######****######
l_1 = 0.00012, 
l_2 = 1 (done) (2 done) 

l_1 = 0.000115, 
l_2 = 1 (now) 2 (done)

l_1 = 0.00011, 
l_2 = 1 (now) (2 now)  



svm (my: 2, others: 1)
################
l_1 = 0.00016, 
l_2 = 1 (42) (2 done) 

l_1 = 0.0002,
l_2 = 1 (42) (2 done)

l_1 = 0.00025,
l_2 = 1 (5 now on margo) , 2 (27) now cpu  (2, 1 not yet)


######****######
l_1 = 0.00016, 
l_2 = 1 (done) (2 done) 

l_1 = 0.0002,
l_2 = 1 (now) (2 now)

l_1 = 0.00025, (both 1, 2 NOW ON POWER$ NEW GEN)

################

crf (my: 2, others: 0) [totl gpu = 2]
################
l_1 = 0.0004,
l_2 = 1 (3 on margo gpu), 2 (21) now cpu (1, 2 not yet)

l_1 = 0.00035,
l_2 = now cpu  (2, 1 not yet)

1 in margo cpu
2 in gru cpu

l_1 = 0.0003, (NOW ON POWER$ NEW GEN)
l_2 = 1 (21) (2 done)


################
l_1 = 0.0004,
l_2 = 1 (done), 2 (done)

l_1 = 0.00035,
l_2 = 2 (done)  1 (done)


l_1 = 0.0003, (NOW ON POWER$ NEW GEN)

l_1 = 0.00025,
l_2 = 1 (now) (2 now)

power4:
################

l_1 = 0.0003,
l_2 = 1 (now) (2 done)

'''

