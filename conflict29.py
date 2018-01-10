

import os, sys, gzip
import time
import math
import json
import cPickle as pickle

from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy as np
import theano
import theano.tensor as T

from nn import create_optimization_updates, get_activation_by_name, sigmoid, linear
from nn import EmbeddingLayer, Layer, LSTM, RCNN, apply_dropout, default_rng, set_default_rng_seed
from utils import say
import myio
import options
from extended_layers import ExtRCNN, ExtLSTM, ZLayer, LZLayer
from nltk.corpus import stopwords

stopWords = [  w.encode('utf-8') for w in list(stopwords.words('english'))+[',', '"', ';', '.', "'s", '(', ')', '-']   ]
# print stopWords
np.random.seed(seed=1111)
total_encode_time = 0
total_generate_time = 0


def get_test_sampled_data(args, sentence_level_sampling_after_recall = 0): #works only when sentence_level_sampling_after_recall != 0
    np.random.seed(seed=1111)
    rationale_data = myio.read_rationales(args.load_rationale)
    rationale_data_ori = myio.read_rationales(args.load_rationale)

    if sentence_level_sampling_after_recall == 0: return rationale_data

    balnk_out_words =[]
    balnk_out_words_num = 0
    total_words = 0

    for i, x in enumerate(rationale_data):
        idx = [ j for j, w in enumerate(x["x"]) if w in '.' ] + [len(x['x']) - 1]
        truez_intvals = x[str(args.aspect)]
        begin = 0
        for end in idx:
            # print ' now begin: ', begin, ' end: ', end
            slngth = end - begin + 1
            total_words += slngth
            if(slngth<1): break
            ratnlngth = 1e-5
            for u in truez_intvals:
                r_b = u[0]
                r_e = u[1] - 1
                
                if(u[0]<begin): r_b = begin
                if(u[1] - 1 >end): r_e = end
                diff = r_e - r_b + 1
                    # print r_b, r_e
                if(diff>0):ratnlngth += diff

            if(  any (end>=u[0] for u in truez_intvals) == False) :
                assert ratnlngth == 1e-5
            elif(  any (begin<u[1]  for u in truez_intvals) == False ):
                assert ratnlngth == 1e-5
            
            rnd = np.random.rand()
            if(ratnlngth<1 and rnd>args.test_smaple_rate):
                # print ' begin: ', begin, ' end: ', end, " true: ", truez_intvals, ' rnd: ', rnd, ' sample rate: ', args.test_smaple_rate
                balnk_out_words_num+=slngth
                balnk_out_words.append(rationale_data[i]['x'][begin:end+1])
                rationale_data[i]['x'][begin:end] = ['<unk>' for j in range(begin, end+1) ]

            begin = end+1

    print " now selection should be: ", (1.00 - (balnk_out_words_num*1.00/total_words)), ' #balnk_out_words: ', balnk_out_words_num, ' total: ', total_words
    
    return rationale_data


def get_train_union_data(args):
    np.random.seed(seed=1111)
    train_x, train_y = myio.read_annotations(args.train)
    train_x_union, train_y_union  = myio.read_annotations(args.train)
    len_train_x_ori = len(train_x_union)

    len_ = int(args.debug*len(train_x))
    train_x_union = train_x_union[0:len_]
    train_y_union = train_y_union[0:len_]

    for i in range(args.num_data):
        for j in range(len_) :
            idx = len(train_x_union)
            train_x_union.append([])
            train_y_union.append(train_y[j])
            x = train_x[j]
            a = 0
            for b in range(len(x)):
                w = x[b]
                if w == '.':
                    rnd = np.random.rand()
                    # if(i==3 and j==3):print i, j, rnd
                    if (rnd>args.balnk_out_prob ): # the reason is at least one sentence should be included
                        for ww in x[a:b+1]: train_x_union[idx].append(ww)
                    a = b+1 

            if len(train_x_union[idx]) == 0:
                train_x_union[idx] = x
    # print 'returning from get union of blanked out dataset'
    # print train_x_union[0]
    # print train_x_union[len_]
    # print train_x_union[2*len_]
    assert len(train_x_union) == (args.num_data+1)*len_
    return train_x_union, train_y_union

def get_sparse(o_x):
    a = np.nonzero(o_x)
    b = [[] for i in range(len(o_x))]
    for n in range(len(a[0])):
        b[a[0][n]].append(o_x[a[0][n]][a[1][n]].astype(int))


    return b


def remove_non_selcted(bx, bz, padding_id):
    assert bx.ndim == bz.ndim == 2
    r = (bz * bx).transpose()
    r[r == padding_id] = 0
    bx_t = get_sparse(r)
    max_len = max(len(x) for x in bx_t)
    return np.column_stack([np.pad(x, (max_len - len(x), 0), "constant", constant_values=padding_id) for x in bx_t])

class Generator(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses


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

        flipped_embs = embs[::-1]

        # len*bacth*n_d
        #h1 = layers[0].forward_all(embs)
        #h2 = layers[1].forward_all(flipped_embs)
        #h_final = T.concatenate([h1, h2[::-1]], axis=2)
        #h_final = apply_dropout(h_final, dropout)
        #size = n_d * 2

        size = n_e


        output_layer = self.output_layer = Layer(
                n_in = size,
                n_out = 1,
                activation = sigmoid
            )

        # len*batch*1 
        #probs = output_layer.forward(h_final)
        probs = output_layer.forward(embs)
    

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



class Encoder(object):

    def __init__(self, args, embedding_layer, nclasses, generator):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        self.generator = generator

    def ready(self):
        global total_encode_time 
        #say("in encoder ready: \n")
        #start_encode_time = time.time()
        generator = self.generator
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = generator.dropout

        # len*batch
        x = generator.x
        z = generator.z_pred
        z = z.dimshuffle((0,1,"x"))

        # batch*nclasses
        y = self.y = T.fmatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]
        depth = args.depth
        layer_type = args.layer.lower()
        for i in xrange(depth):
            if layer_type == "rcnn":
                l = ExtRCNN(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order
                    )
            elif layer_type == "lstm":
                l = ExtLSTM(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            layers.append(l)

        # len * batch * 1
        masks = T.cast(T.neq(x, padding_id).dimshuffle((0,1,"x")) * z, theano.config.floatX)
        # batch * 1
        cnt_non_padding = T.sum(masks, axis=0) + 1e-8

        # len*batch*n_e
        embs = generator.word_embs

        pooling = args.pooling
        lst_states = [ ]
        h_prev = embs
        for l in layers:
            # len*batch*n_d
            h_next = l.forward_all(h_prev, z)
            if pooling:
                # batch * n_d
                masked_sum = T.sum(h_next * masks, axis=0)
                lst_states.append(masked_sum/cnt_non_padding) # mean pooling
            else:
                lst_states.append(h_next[-1]) # last state
            h_prev = apply_dropout(h_next, dropout)

        if args.use_all:
            size = depth * n_d
            # batch * size (i.e. n_d*depth)
            h_final = T.concatenate(lst_states, axis=1)
        else:
            size = n_d
            h_final = lst_states[-1]
        h_final = apply_dropout(h_final, dropout)

        output_layer = self.output_layer = Layer(
                n_in = size,
                n_out = self.nclasses,
                activation = sigmoid
            )

        # batch * nclasses
        preds = self.preds = output_layer.forward(h_final)

        # batch
        loss_mat = self.loss_mat = (preds-y)**2

        pred_diff = self.pred_diff = T.mean(T.max(preds, axis=1) - T.min(preds, axis=1))

        if args.aspect < 0:
            loss_vec = T.mean(loss_mat, axis=1)
        else:
            assert args.aspect < self.nclasses
            loss_vec = loss_mat[:,args.aspect]
        self.loss_vec = loss_vec

        zsum = generator.zsum
        zdiff = generator.zdiff
        logpz = generator.logpz

        coherent_factor = args.sparsity * args.coherent
        loss = self.loss = T.mean(loss_vec)
        sparsity_cost = self.sparsity_cost = T.mean(zsum) * args.sparsity + \
                                             T.mean(zdiff) * coherent_factor
        cost_vec = loss_vec + zsum * args.sparsity + zdiff * coherent_factor
        cost_logpz = T.mean(cost_vec * T.sum(logpz, axis=0))
        self.obj = T.mean(cost_vec)

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

        self.cost_g = cost_logpz * 10 + generator.l2_cost
        self.cost_e = loss * 10 + l2_cost
        #say("finish encoding : {}\n".format(time.time()-start_encode_time))
        #total_encode_time += time.time()-start_encode_time

class Model(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        #print 'in model init: ', self.args.select_all


    def ready(self):
        args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses
        #print 'in model ready: ', args.seed
        self.generator = Generator(args, embedding_layer, nclasses)
        self.encoder = Encoder(args, embedding_layer, nclasses, self.generator)
        self.generator.ready()
        self.encoder.ready()
        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.y = self.encoder.y
        self.z = self.generator.z_pred
        self.word_embs= self.generator.word_embs
        self.params = self.encoder.params + self.generator.params

        #assert len(self.x) == len(self.z)
        #print 'x  shape: ', self.x.shape[0] 
        #print 'z  shape: ', self.z.shape [0]


    def save_model(self, path, args):

        # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        # output to path
        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.encoder.params ],   # encoder
                 [ x.get_value() for x in self.generator.params ], # generator
                 self.nclasses,
                 args                                 # training configuration
                ),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )


    def load_model(self, path, seed = None, select_all = None, load_enc = 0, load_gen = 0): # seed and select all can differ from file
        
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else: path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            eparams, gparams, nclasses, args  = pickle.load(fin)

        # construct model/network using saved configuration
        #self.args = args
        #print self.args.select_all
        if seed is not None: self.args.seed = seed
        if select_all is not None: self.args.select_all = select_all

        self.nclasses = nclasses
        self.ready()
        flag = 0
        if(load_enc ==1):
            print (" loading encoder from ", path)
            for x,v in zip(self.encoder.params, eparams):
                # if(flag<5): 
                #     print 'encoder param: ',v
                x.set_value(v)
                flag+=1
        if(load_gen==1):
            for x,v in zip(self.generator.params, gparams):
                x.set_value(v)

    def load_model2(self, path, path2, seed = None, select_all = None): # seed and select all can differ from file
        
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else: path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            eparams, gparams, nclasses, args  = pickle.load(fin)

        # construct model/network using saved configuration
        #self.args = args
        #print self.args.select_all
        if seed is not None: self.args.seed = seed
        if select_all is not None: self.args.select_all = select_all

        self.nclasses = nclasses
        self.ready()
        flag = 0
        
        print (" loading encoder from ", path)
        for x,v in zip(self.encoder.params, eparams):
            # if(flag<5): 
            #     print 'encoder param: ',v
            x.set_value(v)
            flag+=1

        if not os.path.exists(path2):
            if path2.endswith(".pkl"):
                path2 += ".gz"
            else: path2 += ".pkl.gz"

        with gzip.open(path2, "rb") as fin:
            eparams, gparams, nclasses, args  = pickle.load(fin)

        print (" loading geneartor from ", path2)  
        for x,v in zip(self.generator.params, gparams):
            x.set_value(v)

    # def make_gen_zero(self):
    #     for x in self.generator.params:
    #         print ' x.shape[0], x.shape[1]: ', eval(x).shape[0], eval(x).shape[1]
    #         x.set_value(theano.shared(
    #             np.zeros_like(eval(x).shape[0], eval(x).shape[1]).astype(theano.config.floatX)
    #         ))

    def train(self, train, dev, test, rationale_data, trained_max_epochs = None):
        args = self.args
        args.trained_max_epochs = self.trained_max_epochs = trained_max_epochs
        dropout = self.dropout
        padding_id = self.embedding_layer.vocab_map["<padding>"]



        if dev is not None:
            dev_batches_x, dev_batches_y = myio.create_batches(
                            dev[0], dev[1], args.batch, padding_id
                        )
        if test is not None:
            test_batches_x, test_batches_y = myio.create_batches(
                            test[0], test[1], args.batch, padding_id
                        )
        if rationale_data is not None:
            valid_batches_x, valid_batches_y = myio.create_batches(
                    [ u["xids"] for u in rationale_data ],
                    [ u["y"] for u in rationale_data ],
                    args.batch,
                    padding_id,
                    sort = False
                )

        start_time = time.time()
        train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )
        say("{:.2f}s to create training batches\n\n".format(
                time.time()-start_time
            ))
        updates_e, lr_e, gnorm_e = create_optimization_updates(
                               cost = self.encoder.cost_e,
                               params = self.encoder.params,
                               method = args.learning,
                               beta1 = args.beta1,
                               beta2 = args.beta2,
                               lr = args.learning_rate
                        )[:3]


        # updates_g, lr_g, gnorm_g = create_optimization_updates(
        #                        cost = self.encoder.cost_g,
        #                        params = self.generator.params,
        #                        method = args.learning,
        #                        beta1 = args.beta1,
        #                        beta2 = args.beta2,
        #                        lr = args.learning_rate
        #                 )[:3]


        get_loss_and_pred = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.encoder.loss_vec, self.encoder.preds, self.z ],
                # updates = self.generator.sample_updates
            )

        eval_generator = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.z, self.encoder.obj, self.encoder.loss,
                                self.encoder.pred_diff ],
                # updates = self.generator.sample_updates
            )
        sample_generator = theano.function(
                inputs = [ self.x ],
                outputs = self.z,
                # updates = self.generator.sample_updates
            )
        sample_encoder = theano.function(
                inputs = [ self.x, self.y, self.z],
                outputs = [ self.encoder.obj, self.encoder.loss,
                                self.encoder.pred_diff],
                # updates = self.generator.sample_updates
            )

        
        train_generator_e = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.encoder.obj, self.encoder.loss, \
                                self.encoder.sparsity_cost, self.z, self.word_embs, gnorm_e],
                                # self.encoder.sparsity_cost, self.z, self.word_embs, gnorm_e, gnorm_g, self.generator.probs2],
                updates = updates_e.items() #+ updates_g.items() #+ self.generator.sample_updates,
                # updates = updates_e.items() + updates_g.items() #+ self.generator.sample_updates,
            )
        

        eval_period = args.eval_period
        unchanged = 0
        best_dev = 1e+2
        best_dev_e = 1e+2
        last_train_avg_cost = None
        last_dev_avg_cost = None
        tolerance = 0.001 #+ 1e-3
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        for epoch_ in xrange(args.max_epochs): # -50 when max_epochs  = 100 given
            #print(" max epochs in train func: ", args.max_epochs)
            epoch = args.trained_max_epochs + epoch_
            unchanged += 1
            # if unchanged > 20: 
            #     print 'dev set increases more than 20 times after the best dev found so return'
            #     return

            train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )

            more = True
            if args.decay_lr:
                param_bak = [ p.get_value(borrow=False) for p in self.params ]
                
            start_train_generate = time.time()
            more_counter = 0
            while more:
                processed = 0
                train_cost = 0.0
                train_loss = 0.0
                train_sparsity_cost = 0.0
                p1 = 0.0
                start_time = time.time()

                N = len(train_batches_x)
                #print(" begining : ", train_cost )
                for i in xrange(N):
                    if (i+1) % 100 == 0:
                        say("\r{}/{} {:.2f}       ".format(i+1,N,p1/(i+1)))

                    bx, by = train_batches_x[i], train_batches_y[i]
                    mask = bx != padding_id
                    start_train_time = time.time()
                    cost, loss, sparsity_cost, emb, gl2_e = train_generator_e(bx, by)
                    

                    k = len(by)
                    processed += k
                    train_cost += cost
                    train_loss += loss
                    train_sparsity_cost += sparsity_cost
                    p1 += np.sum(bz*mask) / (np.sum(mask)+1e-8)

                cur_train_avg_cost = train_cost / N
                #print(" end : ", cur_train_avg_cost )
                say("train generate  time: {} \n".format(time.time() - start_train_generate))
                if dev:
                    self.dropout.set_value(0.0)
                    start_dev_time = time.time()
                    dev_obj, dev_loss, dev_diff, dev_p1 = self.evaluate_data(
                            dev_batches_x, dev_batches_y, eval_generator, sampling=True)
                    self.dropout.set_value(dropout_prob)
                    say("dev evaluate data time: {} \n".format(time.time() - start_dev_time))
                    cur_dev_avg_cost = dev_obj

                more = False
                if args.decay_lr and last_train_avg_cost is not None:
                    if cur_train_avg_cost > last_train_avg_cost*(1+tolerance):
                        more = True
                        say("\nTrain cost {} --> {}\n".format(
                                last_train_avg_cost, cur_train_avg_cost
                            ))
                    if dev and cur_dev_avg_cost > last_dev_avg_cost*(1+tolerance):
                        more = True
                        say("\nDev cost {} --> {}\n".format(
                                last_dev_avg_cost, cur_dev_avg_cost
                            ))
                if more:
                    more_counter += 1
                    print('MORE COUNTER: ', more_counter)
                    if more_counter>15 or lr_e.get_value()<1e-10: 
                        print "lr: ", lr_e.get_value(), ' more counter: ', more_counter
                        return
                if more:
                    # more_counter = 0
                    lr_val = lr_e.get_value()*0.5
                    lr_val = np.float64(lr_val).astype(theano.config.floatX)
                    # lr_g.set_value(lr_val)
                    lr_e.set_value(lr_val)
                    say("Decrease learning rate to {} at epoch {}\n".format(float(lr_val),epoch_+1))
                    for p, v in zip(self.params, param_bak):
                        #print ('param restoreing: ', p, v)
                        p.set_value(v)
                    continue

                last_train_avg_cost = cur_train_avg_cost
                if dev: last_dev_avg_cost = cur_dev_avg_cost

                say("\n")
                

                say(("Encoder Epoch {:.2f}  costg={:.4f}  scost={:.4f}  lossg={:.4f}  " +
                    "p[1]={:.2f}  |g|= {:.4f}\t[{:.2f}m / {:.2f}m]\n").format(
                        epoch+(i+1.0)/N,
                        train_cost / N,
                        train_sparsity_cost / N,
                        train_loss / N,
                        p1 / N,
                        float(gl2_e),
                        (time.time()-start_time)/60.0,
                        (time.time()-start_time)/60.0/(i+1)*N
                    ))

                say("\t"+str([ "{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                for x in self.encoder.params ])+"\n")
                say("\t"+str([ "{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                for x in self.generator.params ])+"\n")
                say("total encode time = {} total geneartor time = {} \n".format(total_encode_time, total_generate_time))

                if epoch_ % args.save_every ==0 :#and epoch_>0:
                    print 'saving model after epoch -', epoch_+1, ' file name: ', args.save_model +  str(epoch_)
                    self.save_model(args.save_model+str(epoch_), args)

                if dev:
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        if args.dump and rationale_data:
                            self.dump_rationales(args.dump, valid_batches_x, valid_batches_y,
                                        get_loss_and_pred, sample_generator)

                        if args.save_model:
                            print 'saving best model after epoch -', epoch_ + 1, ' file name: ', args.save_model
                            self.save_model(args.save_model, args)

                    say(("\tsampling devg={:.4f}  mseg={:.4f}  avg_diffg={:.4f}" + 
                                "  p[1]g={:.2f}  best_dev={:.4f}\n").format(
                        dev_obj,
                        dev_loss,
                        dev_diff,
                        dev_p1,
                        best_dev
                    ))

                    if rationale_data is not None:
                        self.dropout.set_value(0.0)

                        start_rational_time = time.time()
                        #r_mse, r_p1, r_prec1, r_prec2 = self.evaluate_rationale(
                        #        rationale_data, valid_batches_x,
                        #        valid_batches_y, eval_generator)


                        
                        r_mse, r_p1, r_prec1, r_prec2, gen_time, enc_time, prec_cal_time, recall, actual_recall = self.evaluate_rationale(
                            rationale_data, valid_batches_x,
                            valid_batches_y, sample_generator, sample_encoder, eval_generator)

                        self.dropout.set_value(dropout_prob)
                        say(("\trationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
                                    "  prec2={:.4f} recall={:.4f} actual recall={:.4f} time nedded for rational={}\n").format(
                                r_mse,
                                r_p1,
                                r_prec1,
                                r_prec2, 
                                recall,
                                actual_recall,
                                time.time() - start_rational_time
                        ))



    def evaluate_data(self, batches_x, batches_y, eval_func, sampling=False):
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        tot_obj, tot_mse, tot_diff, p1 = 0.0, 0.0, 0.0, 0.0
        for bx, by in zip(batches_x, batches_y):
            if not sampling:
                e, d = eval_func(bx, by)
            else:
                mask = bx != padding_id
                bz, o, e, d = eval_func(bx, by)
                p1 += np.sum(bz*mask) / (np.sum(mask) + 1e-8)
                tot_obj += o
            tot_mse += e
            tot_diff += d
        n = len(batches_x)
        if not sampling:
            return tot_mse/n, tot_diff/n
        return tot_obj/n, tot_mse/n, tot_diff/n, p1/n

    def evaluate_rationale(self, reviews, batches_x, batches_y, debug_func_gen, debug_func_enc, eval_func):
        # print "first in evaluate_rational func"
        args = self.args
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        unk_id = self.embedding_layer.vocab_map["<unk>"]
        aspect = str(args.aspect)
        p1, tot_mse, tot_prec1, tot_prec2, tot_rec1 = 0.0, 0.0, 0.0, 0.0, 0.0
        tot_z, tot_n, tot_tn = 1e-10, 1e-10, 1e-10
        cnt = 0
        cnt_t = 0
        cnt_c = 0
        appended_rationales = 0
        actual_rationales = 9725
        stopword_rationales = 8557
        tnum_gold_rationales = 18282 # 9725+8557
        actual_prec = 0
        start_prec_cal_time = time.time()
        encode_total_time = 0
        generate_total_time = 0

        if(args.gen_type == 'rand_word' ):
            print(" rand selection ")
        else: print(" words r selected using gen")


        for bx, by in zip(batches_x, batches_y):
            # print ('one batch')
            mask = bx != padding_id
            
            start_generate_time = time.time()
            # if args.select_all == 1: bz = np.ones_like(bx, dtype=theano.config.floatX) 
            # else:
            #     if args.select_all == 0: bz = np.zeros_like(bx, dtype=theano.config.floatX)
            #     else: bz = debug_func_gen(bx)
            #print 'bx len: ', len(bx), ' bz len: ', len(bz)



            if args.select_all == 1: bz = np.ones_like(bx, dtype=theano.config.floatX) 
            elif args.select_all == 0: bz = np.zeros_like(bx, dtype=theano.config.floatX)
            else:
                    if(args.gen_type == 'rand_word'):
                        print(" rand selection ")
                        np.random.seed(1111)
                        r = np.random.random_sample(bx.shape)
                        bz = r<=args.select_all
                    else: bz = debug_func_gen(bx)

            generator_time = time.time() - start_generate_time
            generate_total_time += generator_time

            start_encode_time = time.time()
            bx_t = np.array(remove_non_selcted(bx, bz,  padding_id)).astype(np.int32) #bx_t has only the words those are selected by the generator
            bz_t = np.ones_like(bx_t, dtype=theano.config.floatX) #so bz_t has all 1 for bx_t


            #o, e, d = debug_func_enc(bx, by, bz) # o, e, d = debug_func_enc(bx, by, bz)
            # print("before encoder: ")
            o, e, d = debug_func_enc(bx_t, by, bz_t)
            # print("after encoder: ")
            encoder_time = time.time() - start_encode_time
            encode_total_time += encoder_time
            


            mask2 = mask #* (bx != unk_id)
            tot_mse += e
            p1 += np.sum(bz*mask2)/(np.sum(mask) + 1e-8)
            if args.aspect >= 0:
                for z,m in zip(bz.T, mask.T):
                    z = [ vz for vz,vm in zip(z,m) if vm ]
                    assert len(z) == len(reviews[cnt]["xids"])
                    truez_intvals = reviews[cnt][aspect]
                    prec = sum( 1 for i, zi in enumerate(z) if zi>0 and \
                                any(i>=u[0] and i<u[1] for u in truez_intvals) )

                    actual_prec += sum( 1 for i, zi in enumerate(z) if zi>0 and reviews[cnt]["x"][i].encode('utf-8')  in stopWords and \
                                any(i>=u[0] and i<u[1] for u in truez_intvals) ) # not in 

                    ntz = sum( u[1] - u[0] for u in truez_intvals) 
                    nz = sum(z)
                    if ntz>0:
                        tot_rec1 += prec/(ntz+0.0)
                        tot_tn +=1
                    # else:
                    #     print (truez_intvals, 'ntz: ', ntz)
                    if nz > 0:
                        tot_prec1 += prec/(nz+0.0)
                        tot_n += 1
                    tot_prec2 += prec
                    tot_z += nz
                    cnt += 1

        #assert cnt == len(reviews)
        n = len(batches_x)
        a_r = actual_prec/(actual_rationales +0.0)
        print('total selection: ', p1, ' total prec1: ', tot_prec1, ' total prec2: ', tot_prec2, ' recall: ' , tot_rec1/tot_tn , 'total gold: ', str(tot_tn), 'actual recall: ', str(actual_prec)+'/'+str(actual_rationales), a_r )
        prec_cal_time = time.time() - start_prec_cal_time
        return tot_mse/n, p1/n, tot_prec1/tot_n, tot_prec2/tot_z, generate_total_time, encode_total_time, prec_cal_time, tot_rec1/tot_tn, a_r #, sum_y/sum_y_counts

    def dump_rationales(self, path, batches_x, batches_y, eval_func, sample_func):
        embedding_layer = self.embedding_layer
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        lst = [ ]
        for bx, by in zip(batches_x, batches_y):
            loss_vec_r, preds_r, bz = eval_func(bx, by)
            assert len(loss_vec_r) == bx.shape[1]
            for loss_r, p_r, x,y,z in zip(loss_vec_r, preds_r, bx.T, by, bz.T):
                loss_r = float(loss_r)
                p_r, x, y, z = p_r.tolist(), x.tolist(), y.tolist(), z.tolist()
                w = embedding_layer.map_to_words(x)
                r = [ u if v == 1 else "__" for u,v in zip(w,z) ]
                diff = max(y)-min(y)
                lst.append((diff, loss_r, r, w, x, y, z, p_r))

        #lst = sorted(lst, key=lambda x: (len(x[3]), x[2]))
        with open(path,"w") as fout:
            for diff, loss_r, r, w, x, y, z, p_r in lst:
                fout.write( json.dumps( { "diff": diff,
                                          "loss_r": loss_r,
                                          "rationale": " ".join(r),
                                          "text": " ".join(w),
                                          "x": x,
                                          "z": z,
                                          "y": y,
                                          "p_r": p_r } ) + "\n" )


def main():
    print args
    set_default_rng_seed(args.seed)
    assert args.embedding, "Pre-trained word embeddings required."

    embedding_layer = myio.create_embedding_layer(
                        args.embedding
                    )

    max_len = args.max_len
    

    if args.train:
        train_x, train_y = get_train_union_data(args) #myio.read_annotations(args.train)
        if args.debug :
            len_ = len(train_x)*args.debug
            len_ = int(len_)
            train_x = train_x[:len_]
            train_y = train_y[:len_]
        print 'train size: ',  len(train_x) #, train_x[0], len(train_x[0])
        #exit()
        train_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in train_x ]


    if args.dev:
        dev_x, dev_y = myio.read_annotations(args.dev)
        if args.debug :
            len_ = len(dev_x)*args.debug
            len_ = int(len_)
            dev_x = dev_x[:len_]
            dev_x = dev_y[:len_]
        print 'train size: ',  len(train_x)
        dev_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in dev_x ]


    if args.load_rationale:

        rationale_data = get_test_sampled_data(args) #myio.read_rationales(args.load_rationale)
        for x in rationale_data:
            x["xids"] = embedding_layer.map_to_ids(x["x"])

    #print 'in main: ', args.seed
    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = len(train_y[0])
                )
        if args.load_model or args.load_gen_only: 
            model.load_model2(args.load_model,args.load_gen_model, seed = args.seed, select_all = args.select_all)
            say("model loaded successfully.\n")
        else:
            model.ready()
            
            # model.make_gen_zero()
        #say(" ready time nedded {} \n".format(time.time()-start_ready_time))

        #debug_func2 = theano.function(
        #        inputs = [ model.x, model.z ],
        #        outputs = model.generator.logpz
        #    )
        #theano.printing.debugprint(debug_func2)
        #return

        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                None, #(test_x, test_y),
                rationale_data if args.load_rationale else None,
                trained_max_epochs = args.trained_max_epochs
            )

    if args.load_model and not args.dev and not args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = -1
                )
        model.load_model2(args.load_model,args.load_gen_model, seed = args.seed, select_all = args.select_all)
        say(" model loaded successfully.\n")


        sample_generator = theano.function(
                inputs = [ model.x ],
                outputs = model.z,
                # updates = model.generator.sample_updates
            )
        sample_encoder = theano.function(
                inputs = [ model.x, model.y, model.z],
                outputs = [ model.encoder.obj, model.encoder.loss,
                                model.encoder.pred_diff],
                # updates = model.generator.sample_updates
            )
        # compile an evaluation function
        eval_func = theano.function(
                inputs = [ model.x, model.y ],
                outputs = [ model.z, model.encoder.obj, model.encoder.loss,
                                model.encoder.pred_diff ],
                # updates = model.generator.sample_updates
            )
        debug_func_enc = theano.function(
                inputs = [ model.x, model.y ],
                outputs = [ model.z, model.encoder.obj, model.encoder.loss,
                                model.encoder.pred_diff ] ,
                # updates = model.generator.sample_updates
            )
        debug_func_gen = theano.function(
                inputs = [ model.x, model.y ],
                outputs = [ model.z , model.encoder.obj, model.encoder.loss,
                                model.encoder.pred_diff],
                # updates = model.generator.sample_updates
            )

        # compile a predictor function
        pred_func = theano.function(
                inputs = [ model.x ],
                outputs = [ model.z, model.encoder.preds ],
                # updates = model.generator.sample_updates
            )

        # batching data
        padding_id = embedding_layer.vocab_map["<padding>"]
        if rationale_data is not None:
            valid_batches_x, valid_batches_y = myio.create_batches(
                    [ u["xids"] for u in rationale_data ],
                    [ u["y"] for u in rationale_data ],
                    args.batch,
                    padding_id,
                    sort = False
                )
        

        # disable dropout
        model.dropout.set_value(0.0)
        if rationale_data is not None:
            #model.dropout.set_value(0.0)
            start_rational_time = time.time()
            r_mse, r_p1, r_prec1, r_prec2, gen_time, enc_time, prec_cal_time, recall, actual_recall = model.evaluate_rationale(
                    rationale_data, valid_batches_x,
                    valid_batches_y, sample_generator, sample_encoder, eval_func)
                    #valid_batches_y, eval_func)

            #model.dropout.set_value(dropout_prob)
            #say(("\ttest rationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
            #            "  prec2={:.4f} generator time={:.4f} encoder time={:.4f} total test time={:.4f}\n").format(
            #        r_mse,
            #        r_p1,
            #        r_prec1,
            #        r_prec2, 
            #        gen_time, 
            #        enc_time,
            #        time.time() - start_rational_time
            #))

            data = str('%.5f' % r_mse) + "\t" + str('%4.2f' %r_p1) + "\t" + str('%4.4f' %r_prec1) + "\t" + str('%4.4f' %r_prec2) \
            +"\t"+str(recall)+"\t"+str(actual_recall)+"\t"+str(recall*r_prec1)+"\t" + str('%4.2f' %gen_time) + "\t" + str('%4.2f' %enc_time) \
            + "\t" +  str('%4.2f' %prec_cal_time) + "\t" +str('%4.2f' % (time.time() - start_rational_time)) +"\t" + str(args.sparsity) \
            + "\t" + str(args.coherent) + "\t" +str(args.max_epochs) +"\t"+str(args.cur_epoch) #+ '\t' + args.gen_type
     
            
            with open(args.graph_data_path, 'a') as g_f:
                print 'writning to file: ', data
                g_f.write(data+"\n")



        

if __name__=="__main__":
    args = options.load_arguments()
    main()

