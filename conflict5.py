
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



# def evaluate_rationale(self, reviews, batches_x, batches_y, debug_func_gen, debug_func_enc, eval_func):
#     #print "first in evaluate_rational func"
#     args = self.args
#     padding_id = self.embedding_layer.vocab_map["<padding>"]
#     aspect = str(args.aspect)
#     p1, tot_mse, tot_prec1, tot_prec2, tot_rec1 = 0.0, 0.0, 0.0, 0.0, 0.0
#     tot_z, tot_n, tot_tn = 1e-10, 1e-10, 1e-10
#     cnt = 0
#     cnt_t = 0
#     cnt_c = 0
#     appended_rationales = 0
#     actual_rationales = 9725
#     stopword_rationales = 8557
#     tnum_gold_rationales = 18282 # 9725+8557
#     actual_prec = 0
#     start_prec_cal_time = time.time()
#     encode_total_time = 0
#     generate_total_time = 0
#     for bx, by in zip(batches_x, batches_y):
#         mask = bx != padding_id
        
#         start_generate_time = time.time()
#         if args.select_all == 1: bz = np.ones_like(bx, dtype=theano.config.floatX) 
#         else:
#             if args.select_all == 0: bz = np.zeros_like(bx, dtype=theano.config.floatX)
#             else: bz = debug_func_gen(bx)
#         #print 'bx len: ', len(bx), ' bz len: ', len(bz)
#         generator_time = time.time() - start_generate_time
#         generate_total_time += generator_time

#         start_encode_time = time.time()
#         bx_t = np.array(remove_non_selcted(bx, bz,  padding_id)).astype(np.int32) #bx_t has only the words those are selected by the generator
#         bz_t = np.ones_like(bx_t, dtype=theano.config.floatX) #so bz_t has all 1 for bx_t


#         #o, e, d = debug_func_enc(bx, by, bz) # o, e, d = debug_func_enc(bx, by, bz)
#         o, e, d = debug_func_enc(bx_t, by, bz_t)
#         encoder_time = time.time() - start_encode_time
#         encode_total_time += encoder_time
        

        
#         tot_mse += e
#         p1 += np.sum(bz*mask)/(np.sum(mask) + 1e-8)
#         if args.aspect >= 0:
#             for z,m in zip(bz.T, mask.T):
#                 z = [ vz for vz,vm in zip(z,m) if vm ]
#                 assert len(z) == len(reviews[cnt]["xids"])
#                 truez_intvals = reviews[cnt][aspect]
#                 prec = sum( 1 for i, zi in enumerate(z) if zi>0 and \
#                             any(i>=u[0] and i<u[1] for u in truez_intvals) )

#                 actual_prec += sum( 1 for i, zi in enumerate(z) if zi>0 and reviews[cnt]["x"][i].encode('utf-8')  in stopWords and \
#                             any(i>=u[0] and i<u[1] for u in truez_intvals) ) # not in 

#                 ntz = sum( u[1] - u[0] for u in truez_intvals) 
#                 nz = sum(z)
#                 if ntz>0:
#                     tot_rec1 += prec/(ntz+0.0)
#                     tot_tn +=1
#                 # else:
#                 #     print (truez_intvals, 'ntz: ', ntz)
#                 if nz > 0:
#                     tot_prec1 += prec/(nz+0.0)
#                     tot_n += 1
#                 tot_prec2 += prec
#                 tot_z += nz
#                 cnt += 1
#     #assert cnt == len(reviews)
#     n = len(batches_x)
#     a_r = actual_prec/(actual_rationales +0.0)
#     print('total selection: ', p1, ' total prec1: ', tot_prec1, ' total prec2: ', tot_prec2, ' recall: ' , tot_rec1/tot_tn , 'total gold: ', str(tot_tn), 'actual recall: ', str(actual_prec)+'/'+str(actual_rationales), a_r )
#     prec_cal_time = time.time() - start_prec_cal_time
#     return tot_mse/n, p1/n, tot_prec1/tot_n, tot_prec2/tot_z, generate_total_time, encode_total_time, prec_cal_time, tot_rec1/tot_tn, a_r #, sum_y/sum_y_counts




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
