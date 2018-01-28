
import sys
import gzip
from gensim.models.keyedvectors import KeyedVectors
import pickle, os
import numpy as np

def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

def load_glove_embedding_iterator(path, dim=300):
    ########### casching the word similarity model (GLOVE), loading  to gensim ###################
    model = None
    if(os.path.isfile('../'+path+".p")):
        model=pickle.load(open('../'+path.split("/")[-1]+'.p','rb'))
    else:
        
        model =  KeyedVectors.load_word2vec_format(path, binary=False)#Word2Vec.load_word2vec_format(path, binary=False)
        pickle.dump(model, open('../'+path+'.p',"wb"))
    return model

