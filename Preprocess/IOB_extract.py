# -*- coding: utf-8 -*-
import numpy as np
import string

redundancy = 2


with open('/home/yigengzhang/workspace/PreProcess/ALLinOne2.txt', 'w') as finaltxt:

	for doc in range(1,22278):
		# print doc

		sentence = np.loadtxt('/home/yigengzhang/workspace/PreProcess/SplitedData/train-' + str(doc) + '.txt', delimiter=' ', dtype = np.str)

		#test with one file
	# sentence = np.loadtxt('/home/yigengzhang/workspace/PreProcess/SplitedData/train-13.txt', delimiter=' ', dtype = np.str)
	# print sentence

		if sentence.ndim > 1:
			(row, col) = sentence.shape

			# if col >= 2:
			# 	# print row, col
				
			# dtype ='a10',at most 10 byte string
			# dtype =object,arbitrary length string, or it comes out as messed up

			tags = np.array(range(row*(col-redundancy)), dtype =object).reshape(row, (col-redundancy))
			pred = np.array(range(row*(col-redundancy)), dtype =object).reshape(row, (col-redundancy))
			ctx_p = np.array(range(row*(col-redundancy)*3), dtype =object).reshape(row, (col-redundancy)*3)
			region_mark = np.array(range(row*(col-redundancy)),dtype=np.int).reshape(row, (col-redundancy))
			region_mark.fill(0)
			# print region_mark


			inbracket = 0

			for j in range(redundancy, col):
				for i in range(row):


					if '(' in sentence[i,j] and ')' in sentence[i,j]:
						tags[i,j-redundancy] = ('B'+'-'+ str(sentence[i,j]).translate(None, string.punctuation))
						# print tags[i,j]
						# print 'B'+'-'+str(sentence[i,j]).translate(None, string.punctuation)



					elif '(' in sentence[i,j] and ')' not in sentence[i,j]:
						tags[i,j-redundancy] = ('B'+'-'+ str(sentence[i,j]).translate(None, string.punctuation))
						beginning = str(sentence[i,j]).translate(None, string.punctuation)
						# print beginning, i, j
						inbracket = 1

					elif sentence[i,j] == '*' and inbracket == 1:
						tags[i,j-redundancy] = ('I'+'-'+ str(beginning))

					elif '(' not in sentence[i,j] and ')' in sentence[i,j]:
						tags[i,j-redundancy] = ('I'+'-'+ str(beginning))
						inbracket = 0

					else:
						tags[i,j-redundancy] = 'O'

			# working on pred, ctx_p and region_mark
						
			(row_t, col_t) = tags.shape
				for j_t in range(col_t):
				for i_t in range(row_t):

					# print ctx_p

					if 'B-V' in tags[i_t,j_t]:

						pred[:,j_t].fill(sentence[i_t,0])

						if i_t == row_t - 1:

							ctx_p[:,j_t*3].fill(sentence[i_t-1,0])
							# print ctx_p[:,j_t*3]
							ctx_p[:,j_t*3+1].fill(sentence[i_t,0])
							# print ctx_p[:,j_t*3+1]
							ctx_p[:,j_t*3+2].fill('。')

							# print 'here'

							region_mark[i_t-1,j_t] = 1
							region_mark[i_t,j_t] = 1


						else:

							ctx_p[:,j_t*3].fill(sentence[i_t-1,0])
							# print ctx_p[:,j_t*3]
							ctx_p[:,j_t*3+1].fill(sentence[i_t,0])
							# print ctx_p[:,j_t*3+1]
							ctx_p[:,j_t*3+2].fill(sentence[i_t+1,0])

							region_mark[i_t-1,j_t] = 1
							region_mark[i_t,j_t] = 1
							region_mark[i_t+1,j_t] = 1

					else:
						pass



			for k in range(col-redundancy):
				np.savetxt(finaltxt,  np.c_[sentence[:,0], pred[:,k], ctx_p[:,k*3:k*3+3],region_mark[:,k], tags[:,k]], delimiter=" ", fmt="%s")
				finaltxt.write('\n')

			# print tags


		if doc%1000 == 0:
			print doc
		else:
			pass
			
	# else:
	# 	pass
		






				

