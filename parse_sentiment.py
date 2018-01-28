import os
# d = 'OTTEN_TOMATOES'
# d= 'IMDB'
# union = 'union_'
# union=''


types = ['JUST_OUTPUT_LAYER']#,'RCNN_RCNN']
for t in types:
	# for union in ['union_','']:
	for union in ['']:
		mydir = '/net/if1/mp5eb/budget_model/'+t+'/CONCATE_NEIGHBOR'
		outfile = '../graph_data/SENTIMENT_full_enc_'+t+'_CONCATE_NEIGHBOR_'+union+'result.txt'

		with open (outfile, 'w') as outf:
			for file in os.listdir(mydir):

				if file.endswith(".txt"):
					# print file, "\n ", t
					if union == '' and file.startswith( 'union' ):
						# print file, "\n ", t
						continue
					elif union == 'union_' and file.startswith( 'model' ):
						# print file, "\n ", t
						continue
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
								if 'mser=' in word: print line
								if 'sparsity=' in word:
									l1= word.split('sparsity=')[1].strip(',')
								if 'coherent=' in word:
									l2= word.split('coherent=')[1].strip(',')
								if 'mser=' in word:
									ac = word.split('mser=')[1].strip(',')
										
								if 'p[1]r=' in word and ac!=-1:
									s = word.split('p[1]r=')[1].strip(',')

								if 'prec1=' in word and ac!=-1:
									e = word.split('prec1=')[1].strip(',')
									
								if 'prec2=' in word and ac!=-1:
									g = word.split('prec2=')[1].strip(',')

								if 'rational=' in word and ac!=-1:
									tt = word.split('rational=')[1].strip(',')
									# is_best = 0
									print str(ac)+"\t"+str(s)+"\t"+str(g)+"\t"+str(e)+"\t"+str(tt)+"\t"+str(l1)+"\t"+str(l2)+"\n"
						
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