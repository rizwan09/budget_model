import os
# d = 'ROTTEN_TOMATOES'
d= 'IMDB'
union_fix = 'union_word'
# union_fix=''
l='dummy_lstm_'
l='dummy_rcnn_'

# types = ['JUST_OUTPUT_LAYER_old','RCNN_RCNN_old']
types = ['LSTM', 'RCNN_RCNN_old']
for t in types:
	if t=='LSTM': l='dummy_lstm_'
	else: l='dummy_rcnn_'
	for union in ['union_words_']:
		mydir = '/net/if1/mp5eb/budget_model/'+d+'/'+t+'/UNION_WORDS/'
		outfile = '../graph_data/'+d+'_'+l+union+'_enc_'+'just_output_layer_gen_result2.txt'


		with open (outfile, 'w') as outf:
			for file in os.listdir(mydir):

				if file.endswith(".txt"):
					# print file, "\n ", t
					if union == '' and file.startswith( 'union' ):
						# print file, "\n ", t
						continue
					elif union == 'union_' and (file.startswith( 'model' )==True or file.startswith( 'union_word' )==True) :
						# print file, "\n ", t
						continue
					elif union == 'union_word' and file.startswith( 'union_word' )==False:
						continue
					# print file, "\n ", t
					with open (os.path.join(mydir, file), 'r') as f:
						l1 = -1
						l2=-1
						is_best = 1
						s =-1
						e=-1
						g=-1
						tt=-1
						ac=-1
						fg = 0
						fe = 0
						dev_found = 0
						dev_s_found = 0
						c =0
						for line in f:

							if 'saving best model' in line:
								is_best = 1 #make 0 again
								c = c+1
								# print line


							for word in line.split():
								# if 'accuracy=' in word: print line
								if 'sparsity=' in word:
									l1= word.split('sparsity=')[1].strip(',')
								if 'coherent=' in word:
									l2= word.split('coherent=')[1].strip(',')
								if 'accuracy=' in word and is_best==1:
									# print line
									if dev_found==0:
										dev_found=1
										# print 'dev found'
									else:
										# dev_found=0
										ac = word.split('accuracy=')[1].strip(',')
										# print word, ac, s, fe, fg, e, g
								if 'p[1]g=' in word and ac!=-1:
									s = word.split('p[1]g=')[1].strip(',')

								if 'time=' in word and dev_found==1 and fg==1 and fe==0 and ac!=-1:
									e = word.split('time=')[1].strip(',')
									fe=1
									continue
								if 'time=' in word and dev_found==1 and fg==0 and fe==0 and ac!=-1:
									g = word.split('time=')[1].strip(',')
									fg=1
									continue
								if 'time=' in word and dev_found==1 and fe==1 and fg==1 and ac!=-1:
									# print word, "\n IN: ", line
								
									dev_found = 0
									tt = word.split('time=')[1].strip(',')
									# is_best = 0
									if(float(s)>0.30): 
										# if (float(ac)>0.871): ac = 0.871
										# print str(ac)+"\t"+str(s)+"\t"+str(g)+"\t"+str(e)+"\t"+str(tt)+"\t"+str(l1)+"\t"+str(l2)+"\n"
						
										outf.write(str(ac)+"\t"+str(s)+"\t"+str(g)+"\t"+str(e)+"\t"+str(tt)+"\t"+str(l1)+"\t"+str(l2)+"\n")
						
									ac = -1
									s = -1
									fg = 0
									fe = 0
								
										

					

						# print str(ac)+"\t"+str(s)+"\t"+str(g)+"\t"+str(e)+"\t"+str(tt)+"\t"+str(l1)+"\t"+str(l2)+"\n"
						
						# if ac!=-1:
						# 	if(float(s)>0.0): 
						# 		# if (float(ac)>0.871): ac = 0.871
						# 		outf.write(str(ac)+"\t"+str(s)+"\t"+str(g)+"\t"+str(e)+"\t"+str(tt)+"\t"+str(l1)+"\t"+str(l2)+"\n")
						
						# print 'coherent=', l2
						# print c
						# print ac
						# print ' writing file complete: ', str(outf)
						# print file
				# break
		print ' writing file complete: ', str(outf)