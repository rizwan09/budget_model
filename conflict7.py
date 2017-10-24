
import os, sys, gzip, string, json
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

stopWords = [  w.encode('utf-8') for w in list(stopwords.words('english'))+  list (string.punctuation) ]#[',', '"', ';', '.', "'s", '(', ')', '-']   ]

# print stopWords
np.random.seed(seed=1111)
total_encode_time = 0
total_generate_time = 0

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

def main():
    print args
    
    embedding_layer = myio.create_embedding_layer(
                        args.embedding
                    )
    padding_id = embedding_layer.vocab_map["<padding>"]
    if args.load_rationale:
        rationale_data = myio.read_rationales(args.load_rationale)
        rationale_data_ori = myio.read_rationales(args.load_rationale)
        for i, x in enumerate(rationale_data):
            # x["xids"] = embedding_layer.map_to_ids(x["x"])
            # print x
            truez_intvals = x[str(args.aspect)]
            # print truez_intvals
            # idx = [0] + [ j for j, w in enumerate(x["x"]) if w in string.punctuation ]
            idx = [ j for j, w in enumerate(x["x"]) if w in '.' ] + [len(x['x']) - 1]
            # print len(x['x']), idx[-1]
            begin = 0
            for end in idx:
                # print ' now begin: ', begin, ' end: ', end
                slngth = end - begin
                if(slngth<1): break
                ratnlngth = 1e-5
                if(args.p>0):
                    for u in truez_intvals:
                        r_b = u[0]
                        r_e = u[1]
                        if(u[0]>= begin or u[1]<=end):
                            if(u[0]<begin): r_b = begin
                            if(u[1]>end): r_e = end+1
                            diff = r_e - r_b
                            # print r_b, r_e
                            if(diff>0):ratnlngth += diff
                    # print ('ratnlngth/slngth: ', ratnlngth/slngth)
                    if(ratnlngth/slngth < args.p):
                        # print ratnlngth, slngth, ratnlngth/slngth, begin, end
                        rationale_data[i]['x'][begin:end] = ["<unk>" for j in range(begin,end+1)]
                else:
                    if(any(begin>=u[0] or end<=u[1] for u in truez_intvals) == False):
                        rationale_data[i]['x'][begin:end] = ["<unk>" for j in range(begin,end+1)]
                begin = end+1
                
            
            

            # break
        # for i, x in enumerate(rationale_data):
        #     print x['x']
        #     print rationale_data_ori[i]['x']
            # break


        file = 'annotations'+str(args.p)+'.json'
        with open(file, 'w') as outfile:
            for i, x in enumerate(rationale_data):
                print x
                outfile.write(json.dumps(x)+"\n")


        # rationale_data_new = myio.read_rationales(file)
        # for i, x in enumerate(rationale_data_new):
        #     print x
        #     break

        # with open('../annotations'+str(args.p)+'.json',"w") as fout:
        #     fout.write( json.dumps(rationale_data))
            




    
        # if rationale_data is not None:
        #     valid_batches_x, valid_batches_y = myio.create_batches(
        #             [ u["xids"] for u in rationale_data ],
        #             [ u["y"] for u in rationale_data ],
        #             args.batch,
        #             padding_id,
        #             sort = False
        #         )
        

        # # disable dropout
        # model.dropout.set_value(0.0)
        # if rationale_data is not None:
        #     #model.dropout.set_value(0.0)
        #     start_rational_time = time.time()
        #     r_mse, r_p1, r_prec1, r_prec2, gen_time, enc_time, prec_cal_time, recall, actual_recall = model.evaluate_rationale(
        #             rationale_data, valid_batches_x,
        #             valid_batches_y, sample_generator, sample_encoder, eval_func)
        #             #valid_batches_y, eval_func)




        

if __name__=="__main__":
    args = options.load_arguments()
    main()
