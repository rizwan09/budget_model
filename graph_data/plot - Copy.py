import argparse
import os
import numpy as np


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
# plt.style.use('bmh')


def plot_graph(x, y, y_label, g_label, name, args):
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    plt.plot(x, y, label = g_label)
    plt.xlabel('Percentage')
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.ylabel(y_label)
    plt.legend()
    loss_fname = os.path.join(args.graph_data_folder, name+'.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data_path",
            type = str,
            #default = 'graph_data/data_vs_lamda_table _Copy.txt'
            default = 'graph_data/data_vs_lamda_table_dummy.txt'
        )
    parser.add_argument("--graph_data_folder",
            type = str,
            default = 'graph_data'
        )
    args = parser.parse_args()

    
    trainData = np.loadtxt(args.graph_data_path, delimiter='\t')
    r_mse, r_p1, r_prec1, r_prec2, gen_time, enc_time, tmp_t, total_test_time = np.split(trainData, [1, 2, 3, 4, 5, 6, 7], axis=1)
    #with open (args.graph_data_path, 'r') as data_f:
        #for line in data_f:
            #for word in line.split('\t'):
                #print word
    r_p1 =  np.concatenate(np.dot(r_p1, 100), axis=0) 
    r_mse = np.concatenate(r_mse, axis=0) 
    gen_time =  np.concatenate(gen_time, axis=0)
    enc_time = np.concatenate(enc_time, axis = 0)
    total_test_time = np.concatenate(total_test_time, axis = 0)
    print r_mse, r_p1

    plot_graph(r_p1, r_mse, y_label = 'MSE', g_label = 'loss vs selection', name = 'loss_mse', args = args)
    plot_graph(r_p1, gen_time, y_label = 'Generation time', g_label = 'Generation time vs selection', name = 'Generation_time', args = args)
    plot_graph(r_p1, enc_time, y_label = 'Encoder time', g_label = 'Encoder time vs selection', name = 'Encoding_time', args = args)
    #plt.plot(r_p1, gen_time, label = 'Genration time vs percentage selection')
    #plt.plot(r_p1, enc_time, label = 'Encoder time vs percentage selection')
    #plt.plot(r_p1, total_test_time, label = 'Total time vs percentage selection')

    


    
    

if __name__ == '__main__':
    main()
