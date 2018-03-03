import os

lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075, 0.0008, 0.00085, 0.0009, 0.001, 0.0001, -1, -2, -3]
# lamda_1 = lamda_1[::-1]


# lamda_2 = [0.01, 0.05, 0.1, -4, -5, -3, -2, -1, -.5, -0.1, 0, 0.25, 0.75, 0.85, 0.95, -0.25, -0.75, -0.85, -0.95, 1,-2,0.5, 0.75, -2.75, -3, 5.5, -6, -6.5]



lamda_2 = [0, 0.25, 0.75, 0.85, 0.95, -0.25, -0.75, -0.85, -0.95]
lamda_2 = [1, 0.5, 0.75, -2.75, -3, 5.5, -6, -6.5] #, 0.1,  0.2]
lamda_2 = [0.01, 0.05, 0.1, -4, -5,  -1, -.5, -0.1]



# lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 0.00065, 0.0007, 0.00075, 0.0008, 0.00085, 0.0009, 0.001, 0.001, 0.0001, -1, -2, -3]
# lamda_1 = lamda_1[::-1]

# lamda_1 = [-3]


# lamda_2 = [-2.75]


# lamda_1 = [0.0001]

# lamda_2 = [-5]






dropout = [0.1]
dp = 0.1
trained_max_epochs = 0
load_emb_only = 1
max_epochs = 100
# learning = 'sgd'
aspect = 1
debug = 1
select_all = 1
output_file = 'rough2_new_gen_outputs_'+"lstm_0.01"+'.json'
covered_percentage = []
# graph_data_file = '../graph_data/data_vs_lamda_table_dummy.txt'
# open(graph_data_file, 'w')
union='union_words_'
# union=''
num_data=0

d = 128
batch_size  = 128
# _type = '../IMDB/LSTM/'
_type = '../IMDB/'



### change param
l='lstm'
l='rcnn'




tt='LSTM'
if l=='rcnn':
	tt='RCNN_RCNN_old'
if union=='union_words_':
	d=200
	batch_size=256
f = 0
for l_1 in lamda_1:
	# f+=1
	# if(f>4):exit()
	for l_2 in lamda_2:
		for lr in [ 0.0005]:
			
			l2_reg=1e-6
			dpx=0.1
	
			# pyfile = 'lstm_gen_lstm_enc_imdb.py'

			if union=='union_words_':
				dp = 0.1
				lr=0.0005

			pyfile = 'just_output_layer_lstm_enc_imdb.py'
			# load_model_file = 'model_'+l+'_sparsity_'+str(0)+'_coherent_'+str(0)+'_dropout_'+str(dpx)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'_batch_'+str(batch_size)+'_d_'+str(d)+'.txt.pkl.gz'#+'_depth_'+str(depth)+'.txt.pkl.gz'
			load_model_file = l+'_model_sparsity_'+str(0)+'_coherent_'+str(0)+'_dropout_'+str(dpx)+"_lr_"+str(lr)+'_full_trainset_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'#+'_depth_'+str(depth)+'.txt.pkl.gz'
			
			model_file = 'model_sparsity_'+str(l_1)+'_coherent_'+str(l_2)+'_dropout_'+str(dp)+"_lr_"+str(lr)+'_max_epochs_'+str(max_epochs)+'.txt.pkl.gz'
			run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" python '+pyfile+' --trained_max_epochs '+str(trained_max_epochs) +' --max_epochs '+ str(max_epochs) +'  --train imdb --dev imdb --test imdb  --embedding glove.6B.300d_w_header.txt' + \
				' --dump ' + output_file +' --sparsity ' + str(l_1) +' --coherent ' + str(l_2) + ' --dropout '+ str(dp)+' --debug '+ str(debug) +' --select_all ' +str(select_all) + ' --learning_rate '+str(lr)+' --save_model ' +_type+ tt+"/UNION_WORDS/MODELS/"+union+model_file \
				+ ' --load_model ' + _type +'RCNN_RCNN_old/CONFLICT42/MODELS/'+union+load_model_file +' --layer '+l+' --batch '+str(batch_size)+' -d '+str(d) + ' --num_data '+str(num_data)#+ ' --load_gen_model ' + _type +'MODELS/'+union+load_model_file
			
			#run_command = 'python generator_fix.py --embedding word_vec.gz --load_rationale annotations.json --dump '+output_file+' --select_all ' +str(select_all)+ ' --aspect ' +str(aspect) +' --sparsity '+str(l_1)+' --coherent '+str(l_2)+' --load_model ' + 'model_new_generators/'+model_file #+ ' --graph_data_path '+ graph_data_file
			
			run_command+= ' >> '+ _type+tt+'/UNION_WORDS/'+union+model_file + '.txt' 	
			print run_command
			os.system(run_command)
			print '\n\n\n'
			# exit()

# exit()
# select_all
# train, dev

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

