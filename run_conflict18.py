import os

<<<<<<< HEAD
lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]

lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]

# lamda_1 = [ 0.00011, 0.000115,  0.00012,  0.00016]
=======
# lamda_1 = [0.000085, 0.000095, 0.0001, 0.000105,  0.00011, 0.000115,  0.00012,  0.00016, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004 ]

# lamda_1 = [ 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0005 ]

lamda_1 = [ 0.0003]
>>>>>>> b2a76fce817641dd735110dfa30604eb8e4c0f75
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

_type = '../JUST_OUTPUT_LAYER/NEIGHBOR/'
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
						
						
						py_file = 'conflict18.py'

						run_command = ' THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" python '+ py_file  +' --max_epochs '+ str(max_epochs) +' --embedding ../word_vec.gz --train ../reviews.aspect1.train.txt.gz --dev ../reviews.aspect1.heldout.txt.gz  --load_rationale ../annotations.json --aspect ' + str(aspect) + \
<<<<<<< HEAD
						' --dump ' + output_file + ' --sparsity ' + str(l_1) +' --coherent ' + str(l_2)+' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						+ ' --learning_rate '+str(lr)  +' --save_model ' + _type +'MODELS/'+model_file #+ ' --debug 0.001' 
						
						run_command+= ' >> '+_type+model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						# exit()
=======
						' --dump ' + output_file + ' --dropout '+  str(dp) +' --debug '+ str(debug) +' --select_all ' +str(select_all) \
						+ ' --learning_rate '+str(lr)  +' --save_model ' + _type +'MODELS/'+model_file + ' --debug 0.001' 
						
						# run_command+= ' >> '+_type+model_file +'.txt'
						print run_command 
						os.system(run_command)
						print '\n\n\n'
						exit()
>>>>>>> b2a76fce817641dd735110dfa30604eb8e4c0f75
	# exit()
'''
class Generator(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        self.nneighbor = 2 #context of 5 words

    def get_avg_emb(self, embs_mid, embs, i):
        return T.mean(self.word_embs[i-self.nneighbor: i+self.nneighbor+1,:,:], axis = 0), i+1 #theano.scan_module.until(embs_mid)

    def ready(self):
        global total_generate_time
        #say("in generator ready: \n")
        #start_generate_time = time.time()
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # len*batch
        x = self.x = T.imatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)
        
        layers = self.layers = [ ]
        # len * batch
        #masks = T.cast(T.neq(x, padding_id), theano.config.floatX)
        masks = T.cast(T.neq(x, padding_id), theano.config.floatX ).dimshuffle((0,1,"x"))

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)
        self.word_embs = embs

        # flipped_embs = embs[::-1]
        # average( word_embeddings(word_t-2,t-1,t,t+1,t+2))

        # len*batch*n_e
        avg_embs = T.zeros((x.shape[0], x.shape[1], n_e))

         #for negative indexing its hould start from -1

        for i in range(self.nneighbor):
            avg_embs = T.set_subtensor(avg_embs[i, :, :],T.mean(embs[i:self.nneighbor+1,:,:], axis = 0)) #along the len dimention
        [avg_emb_middle, i], _ = theano.scan(
                    fn = self.get_avg_emb,
                    sequences = embs[self.nneighbor:-self.nneighbor],
                    outputs_info = [ embs[0,:,:], self.nneighbor]
                )
        avg_embs = T.set_subtensor( avg_embs[self.nneighbor:-self.nneighbor:], avg_emb_middle)
        for i in range(self.nneighbor):
            avg_embs = T.set_subtensor(avg_embs[-i-1, :, :],T.mean(embs[-i-1-self.nneighbor:,:,:], axis = 0))
        # len*bacth*n_d
        #h1 = layers[0].forward_all(embs)
        #h2 = layers[1].forward_all(flipped_embs)
        #h_final = T.concatenate([h1, h2[::-1]], axis=2)
        #h_final = apply_dropout(h_final, dropout)
        #size = n_d * 2 
        ## ans_ = T.mean(embs[index-self.nneighbor: index+self.nneighbor+1 ,:,:], axis = 0)

        size = n_e


        output_layer = self.output_layer = Layer(
                n_in = size,
                n_out = 1,
                activation = sigmoid
            )

        # len*batch*1 
        # probs = output_layer.forward(embs)
        probs = output_layer.forward(avg_embs)
    

        # len*batch
        probs2 = self.probs2 = probs.reshape(x.shape)
        if self.args.seed is not None: self.MRG_rng = MRG_RandomStreams(self.args.seed)
        else: self.MRG_rng = MRG_RandomStreams()
        z_pred = self.z_pred = T.cast(self.MRG_rng.binomial(size=probs2.shape, p=probs2), theano.config.floatX) #"int8")

        # we are computing approximated gradient by sampling z;
        # so should mark sampled z not part of the gradient propagation path
        #


        z_pred = self.z_pred = theano.gradient.disconnected_grad(z_pred)
        #self.sample_updates = sample_updates
        print "z_pred", z_pred.ndim

        z2 = z_pred.dimshuffle((0,1,"x"))
        logpz = - T.nnet.binary_crossentropy(probs, z2) * masks
        logpz = self.logpz = logpz.reshape(x.shape)
        probs = self.probs = probs.reshape(x.shape)

        # batch
        z = z_pred
        self.zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
        self.zdiff = T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)

        params = self.params = [ ]
        for l in layers + [ output_layer ]:
            for p in l.params:
                params.append(p)
        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                                        for x in params)
        say("total # parameters: {}\n".format(nparams))

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p**2)
            else:
                l2_cost = l2_cost + T.sum(p**2)
        l2_cost = l2_cost * args.l2_reg
        self.l2_cost = l2_cost
        #say("finish generating : {}\n".format(time.time()-start_generate_time))
        #total_generate_time += time.time()-start_generate_time
'''

