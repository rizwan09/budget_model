import os, sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_mse_data_graph( y, g_label):
	plt.ylim(0.02, 0.024)
	plt.plot(y, label = g_label)
    

def getAllFiles(root,fileext):
    # print ('root: ', )
    # with open ('tests/table_'+root+'.txt', 'w') as f:

        filepaths = []
        for base, dirs, files in os.walk(root):
            # print ('base, dirs: ', base, dirs)
            for file in files:
                if file.endswith(fileext):
                    print "__"*89+'\n'
                    print (file)
                    print "__"*89+'\n'
                    costg = []
                    mse = []
                    with open ('../graph_data/parsed_table_'+folder+'.txt', 'w') as f, open (os.path.join(base, file), 'r') as r_f:
                    	# f.write(os.path.join(base, file) + '\n')costg=
                    	for line in r_f:
                    		for word  in line.split(" "):
                    			if word.startswith('costg='):
                    				# print 'costg: '+word.split("=")[1]+"\n"
                    				costg.append(float(word.split("=")[1]))
                    			if word.startswith('mser='):
                    				# print 'MSE: '+word.split("=")[1]+"\n"
                    				mse.append(float(word.split("=")[1]))
                    print "__"*89+'\n'
                    # print "costg: ", costg
                    # print "mse: ", mse
                    for i in range(len(costg)):
                    	print 'epoch: ', i , ' train cost: ', costg[i], ' test mse: ', mse[i]
                    

                    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
                    plt.xlabel('epoch')
                    plt.ylabel("COSTG")
                    plt.legend()
                    plot_mse_data_graph(costg, 'costg')
                    loss_fname = os.path.join(base, './train_graph/'+file+'_costg_vs_total_time.png')
                    plt.savefig(loss_fname)
                    print('Created {}'.format(loss_fname))

                    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
                    plt.xlabel('epoch')
                    plt.ylabel("mse")
                    plt.legend()
                    plot_mse_data_graph(mse, 'mse')
                    loss_fname = os.path.join(base,'./train_graph/'+file+'_mse_vs_total_time.png')
                    plt.savefig(loss_fname)
                    print('Created {}'.format(loss_fname))

                    print "__"*89+'\n'
                    # exit()

folder = 'FIX_ENC_TRAIN_RCNN_GEN2' 
folder = 'RSGEN_PREV' 

getAllFiles('../JUST_OUTPUT_LAYER/'+folder, 'txt')