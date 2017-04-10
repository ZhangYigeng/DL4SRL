# -*- coding: utf-8 -*-
import numpy as np
# import string

# redundancy = 2
with open('/home/yigengzhang/workspace/PreProcess/labels.txt', 'w') as labeltxt:
# def loadlabel():

	text = np.loadtxt('/home/yigengzhang/workspace/PreProcess/ALL_no_n.txt', delimiter=' ', dtype = np.str)
	labels = np.unique(text[:,1])
	# print labels
	# print labels.size
	num_labels = range(1,labels.size+1)
	# label_dict = dict(
	# 	zip(labels, range(1,labels.size+1))
	# 	)
	label_dict = np.column_stack((labels,num_labels))

	np.savetxt(labeltxt, label_dict, delimiter=" ", fmt="%s")

# print label_dict