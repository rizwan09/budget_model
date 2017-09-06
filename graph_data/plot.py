import argparse
import os
import numpy as np


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
# plt.style.use('bmh')

types = ['taolei_rcnn_rcnn', 'rcnn_rcnn', 'linear_rcnn', 'just_output_layer']
full_enc_time = []
full_enc_mse = []
full_enc_time.append(1.86 )# with l_1 = 0.000105, l_2 = 1
full_enc_mse.append(0.01231)
full_enc_time.append(0.70) #0.01055    1.00    0.1466  0.1450  0.00    0.70    1.03    1.03    0.0003  1.0 100 -1
full_enc_mse.append(0.01055)
full_enc_time.append(0.70) #0.01151 1.00    0.1466  0.1450  0.00    0.70    1.04    1.04    0.000115    2.0 100 -1
full_enc_mse.append(0.01151)
full_enc_time.append(0.70) #0.01081 1.00    0.1466  0.1450  0.00    0.69    1.03    1.03    0.000105    1.0 100 -1 LINEAR_RCNN
full_enc_mse.append(0.01081)

parser = argparse.ArgumentParser()
parser.add_argument("--graph_data_path",
        type = str,
        #default = 'graph_data/data_vs_lamda_table _Copy.txt'
        default = '../graph_data/data_vs_lamda_table_dummy.txt'
        #default = '../graph_data/data_vs_lamda_table_3Par.txt'
    )
parser.add_argument("--graph_data_folder",
        type = str,
        default = '../graph_data'
    )
args = parser.parse_args()
color = ['r', 'g', 'b', 'm', 'c', 'y', 'k']

#shape_full= ['r*', 'gv', 'b^', 'y<', 'c>' ]
shape_full= ['r*', 'g*', 'b*', 'y*', 'c*', ]
shapes = ['8', 's', 'p', 'o', 'h', 'H', 'D', 'd', 'P', 'X']
#shape = ['-gD', '-b>','-ro', '-y<']



def plot_gen_data_graph(x,g_label, s):
    print "in gen data graph: ", g_label
    if g_label.startswith('full_enc_time'):
        plt.plot([0, 50], [x,x], s, label = g_label) # 50 is manually ploted 
    else: plt.plot(x, s, label = g_label)
    #plt.xlabel('time (seconds)')
    

def plot_mse_data_graph(x, y, g_label,  s):
    if g_label.endswith('rcnn') == False:
        g_label+='_rcnn'
    plt.plot(x, y, s, label = g_label)
    
    
    



avg_gen_time = []

def main(time = 1):
    
    def routine(f, g_label, s):
    
        #f = args.graph_data_path + path
        trainData = np.loadtxt(f, delimiter='\t')
        r_mse, r_p1, r_prec1, r_prec2, gen_time, enc_time, tmp_t, total_test_time, l_1, l_2, m_e, e = np.split(trainData, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], axis=1)
        
        size = []
                   
        
       
        r_p1 =  np.concatenate(np.dot(r_p1, 100), axis=0) 
        r_mse = np.concatenate(r_mse, axis=0) 
        gen_time =  np.concatenate(gen_time, axis=0)
        enc_time = np.concatenate(enc_time, axis = 0)
        total_test_time = np.concatenate(total_test_time, axis = 0)
        l_1 = np.concatenate(l_1, axis = 0)
        l_2 = np.concatenate(l_2, axis = 0)
        e =  np.concatenate(e, axis = 0)
        
        assert len(gen_time) == len(enc_time) 

        gen_enc_time = []
        avg = np.mean(gen_time)
        
        for i in range(len(gen_time)):
            gen_enc_time.append(gen_time[i]+enc_time[i])

        assert len(gen_time) == len(enc_time) == len(gen_enc_time)
        #print 'r_mse, total_time: ', r_mse, gen_enc_time, ' avg gen time: ',avg
        if(time==1):plot_mse_data_graph( gen_enc_time, r_mse,  g_label, s)
        if time == 0: plot_mse_data_graph( r_p1, r_mse,  g_label, s)
        if time== 2: plot_gen_data_graph( gen_time,  g_label, '-'+s)
        avg_gen_time.append(avg)



    



    for i in range(len(types)):
        #print full_enc_time[i], full_enc_mse[i],  types[i], color[i]+'*'
        if(time==1):plot_mse_data_graph( full_enc_time[i], full_enc_mse[i],  "full_enc_"+types[i], color[i]+'*')
        if time == 0: plot_mse_data_graph( 100.00, full_enc_mse[i],  "full_enc_"+types[i], color[i]+'*') # selection = 100% for full enc
        if time==2 and i<2: 
            #print " goin to gen data graph"
            if(i==0): 
                g_label = "full_enc_time_of_taolei_rcnn_rcnn"
                c = '--r*'
            else:
                g_label = "full_enc_time_of_other_models"
                c = '--k*'

            plot_gen_data_graph( full_enc_time[i] , g_label , c)
        if(i>0):routine('../graph_data/'+"tabel_"+types[i].upper()+"_-1.txt", types[i], color[i]+shapes[i])
    #print 'avg_gen_time: ', avg_gen_time

    

    

if __name__ == '__main__':
    #time = 1 for mse_vs_total_time, 0 for mse_vs_selection, 2 for gen_times_of_diff_models
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    main(time = 1)
    plt.xlabel('time (seconds)')
    plt.ylabel("Loss (MSE)")
    plt.legend()
    loss_fname = os.path.join(args.graph_data_folder, 'mse_vs_total_time.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

    avg_gen_time = []
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    main(time = 0)
    plt.xlabel('% rationale selection ')
    plt.ylabel("Loss (MSE)")
    plt.legend()
    loss_fname = os.path.join(args.graph_data_folder, 'mse_vs_selection.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))


    avg_gen_time = []
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    main(time = 2)
    plt.ylabel("Genration time (seconds)")
    plt.legend()
    loss_fname = os.path.join(args.graph_data_folder, 'generation_time.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))



    