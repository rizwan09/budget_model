


    
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




def main():
    print args
    
    # embedding_layer = myio.create_embedding_layer(
    #                     args.embedding
    #                 )
    # padding_id = embedding_layer.vocab_map["<padding>"]
    if args.train:
        np.random.seed(seed=1111)
        train_x, train_y = myio.read_annotations(args.train)
        train_x_union, train_y_union  = myio.read_annotations(args.train)
        len_train_x_ori = len(train_x_union)

        for i in range(args.max_epochs):
            for j in range(len(train_x)) :
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
                        if (rnd>args.balnk_out_prob or b+1<len(x)): # the reason is at least one sentence should be included
                            for ww in x[a:b+1]: train_x_union[idx].append(ww)
                        a = b+1 


        #     assert len(train_x_union) == len(train_y_union)

        print("ori: \n", train_x[-1], train_y[-1], "----"*10)
        print('new after 1st iter: \n', train_x_union[2*len_train_x_ori-1],  train_y_union[2*len_train_x_ori-1], "----"*10)
        print('new after 3nd iter: \n', train_x_union[4*len_train_x_ori-1],  train_y_union[4*len_train_x_ori-1], "----"*10)
        print('final: \n', train_x_union[-1],  train_y_union[-1], "----"*10)
        print('ori size: ', len_train_x_ori)
        print(" final length: ",len(train_x_union) ,  len(train_y_union) )

if __name__=="__main__":
    args = options.load_arguments()
    if(args.p<=0):args.p = 0.5
    main()

