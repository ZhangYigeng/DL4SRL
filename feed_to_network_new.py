# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import numpy as np
import gensim
import itertools
from itertools import izip_longest
from itertools import islice

import time

import tensorflow as tf

# directories:
path_word2vec_model = '/home/yigengzhang/workspace/Word2Vec/wiki.zh.text.new.model'
path_label_dictionary = '/home/yigengzhang/workspace/PreProcess/labels_dictionary.txt'
path_data_to_process = '/home/yigengzhang/workspace/PreProcess/AllinOne/training_dataset_Sample_22sentences' # 22 sentences in total
#path_data_to_process = '/home/yigengzhang/workspace/PreProcess/AllinOne/training_dataset_Sample _10sentences'  # 10 sentences in total 
path_data_to_test = '/home/yigengzhang/workspace/PreProcess/AllinOne/training_dataset_Sample_22sentences'
path_logfile = '/home/yigengzhang/workspace/PreProcess/log'


# load word vector model
model = gensim.models.Word2Vec.load(path_word2vec_model)

# parameters
vec_length = 400
max_sentence_length = 60
padding_vector_x = [0.]*2001
padding_vector_y = [0.]*69

display_step = 1000


# construct a label dictionary
def label_dic():
	dictionary = np.loadtxt(path_label_dictionary, delimiter=' ', dtype = np.str)
	label_dictionary = dict(zip(dictionary[:,0], dictionary[:,1]))
	return label_dictionary

# separate a text chunk using an empty line
def isa_group_separator(line):
	return line == '\n'

# look up one word from a word vector embedding model
def look_up(word): 		
	word = unicode(word, 'utf-8')
	try:
		return model[word]
	except KeyError:
		# here it should be a list instead of a numpy array
		no_vector = [0.0] * vec_length
		# print "not in vocabulary"
		# print no_vector
		return no_vector

# construct vector for each word in sentence (argu,pred,ctx_p_1,2,3)
def construct_vector(sequence):
	vector = []
	# look up each word
	for i in range(5):
		vector.append(look_up(sequence[i]))

	# append region mark
	vector.append([float(sequence[5])])
	represent_as_vector = np.array(vector)

	return represent_as_vector


def generate_vector(one_line):

	sequence = np.asarray(one_line.strip().split(" "))
	# vectorization |argu|pred|ctx1|ctx2|ctx3|
	x_vector = construct_vector(sequence)
	# label for each sentence
	label = int(label_dictionary[sequence[6]])
	y_vector = np.zeros(69)
	y_vector[label-1] = 1.0

	return x_vector, y_vector


def padding(x_batch, y_batch, length_of_current_sentence):

	num_to_padding = max_sentence_length - length_of_current_sentence

	padding_to_x_element, padding_to_y_element = np.array(padding_vector_x), np.array(padding_vector_y)

	x_input = np.vstack(
		(x_batch, np.tile(padding_to_x_element, (num_to_padding, 1)))
		)
	y_input = np.vstack(
		(y_batch, np.tile(padding_to_y_element, (num_to_padding, 1)))
		)

	return x_input, y_input


def div_igonre0(a, b):
	""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide( a, b )
		c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
	return c



# Network parameters
lstm_size = 10
batch_size = max_sentence_length
data_type = tf.float32
label_dictionary = label_dic()

# Network Parameters
n_steps = 2001 # timesteps # features
n_classes = 69 # total classes (0-9 digits)

with tf.name_scope("inputs"):
	x = tf.placeholder("float", [batch_size,n_steps,1],name = "x_inputs")
	y = tf.placeholder("float", [batch_size, n_classes],name = "y_inputs")



def RNN(x):

	with tf.name_scope("RNN"):

		with tf.name_scope("softmax_layer"):
			softmax_w = tf.Variable(tf.random_normal([lstm_size, n_classes], stddev=0.35), name="softmax_w")
			tf.summary.histogram('softmax_w',softmax_w)
			softmax_b = tf.Variable(tf.random_normal([n_classes], stddev=0.35), name="softmax_b")			
			tf.summary.histogram('softmax_b',softmax_b)

		x = tf.unstack(x, n_steps, 1)

		# try
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
		output, state = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		with tf.name_scope("Wx_plus_b"):
			predict_label = tf.matmul(output[-1], softmax_w) + softmax_b
			tf.summary.histogram('network_output',predict_label)

		return predict_label


# file = path_data_to_process
def generate_input(group):

	# print(key,list(group))  # uncomment to see what itertools.groupby does.
	x_batch = []
	y_batch = []

	# iterobj can only be used once and then vanishes! Notice!

	for one_line in group:

		x_vector, y_vector = generate_vector(one_line)

		# Making a flat list out of list of lists, to flattern x_vector:
		# 6 features together, unpack |400|400|400|400|400|1|, 2001 dimensions
		x_merged_vector = list(itertools.chain.from_iterable(x_vector))
		x_batch.append(x_merged_vector)
		# should look like: [0 0 0 ... 1 ...0]
		y_batch.append(y_vector)

	length_of_current_sentence = len(y_batch)

	'''			'''
	''' Padding '''
	'''         '''
	# if < max_sentence_length, padding; else cut
	if length_of_current_sentence <= max_sentence_length:

		x_input, y_input = padding(x_batch, y_batch, length_of_current_sentence)
		x_input = x_input.reshape((max_sentence_length, 2001, 1))
		
	else:
	
		x_input = x_batch[:max_sentence_length]
		y_input = y_batch[:max_sentence_length]

		x_input = np.array(x_input)
		y_input = np.array(y_input)


		x_input = x_input.reshape((max_sentence_length, 2001, 1))

	return x_input, y_input, length_of_current_sentence



with tf.name_scope("Prediction"):
	pred = RNN(x)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(real_label * tf.log(test_label), reduction_indices=[1]))
with tf.name_scope("Loss_cross_entropy"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = pred))
	tf.summary.scalar('loss_cross_entropy',cross_entropy)

with tf.name_scope("Gradient_Descent_Optimizer"):
	# learning rate: 0.01, 0.05
	optimizer = tf.train.RMSPropOptimizer(0.05)
	# optimizer = tf.train.GradientDescentOptimizer(0.01)

with tf.name_scope("Train"):
	train = optimizer.minimize(cross_entropy)

with tf.name_scope("Argmax_pred"):
	argmax_pred = tf.argmax(pred,1)

with tf.name_scope("Correct_pred"):
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

with tf.name_scope("Correct_pred_index"):
	correct_pred_index = tf.cast(correct_pred, tf.float32)

with tf.name_scope("Accuracy"):
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# save variables
saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:

	sampling_sentence_num = 22
	epoch_num = 100

	# statistic for loss
	statistical_matrix = np.zeros((sampling_sentence_num, epoch_num))
	#
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(path_logfile + '/train',sess.graph)
	''' tensorboard command: tensorboard --logdir='0607/'  '''
	sess.run(init)

	# print "initialized!!!"

	'''			 '''
	''' Training '''
	'''			 '''

	### Epoch: ###
	# one epoch = one forward pass and one backward pass of all the training examples, now only forward
	for epoch in range(epoch_num):

		line_num = 0
		count = 0

		with open(path_data_to_process) as f:

			# every sentence construct a batch
			for key,group in itertools.groupby(f,isa_group_separator):
	
				if not key:
				
					x_input, y_input, length_of_current_sentence = generate_input(group)
		
					'''Run OP'''
					sess.run(train, feed_dict={x: x_input, y: y_input})

					##### for large dataset
					# sampling n th sentence
					# if count in [2000, 4000, 6000, 8000, 10000, 20000]:

					# 	# acc = sess.run(accuracy, feed_dict={x: x_input, y: y_input})
					# 	# Calculate batch loss
					# 	loss = sess.run(cross_entropy, feed_dict={x: x_input, y: y_input})
					# 	# print "Iter " + str(count * display_step) + ":"
					# 	# print " Minibatch Loss= "
					# 	# print loss
					# 	# print "Training Accuracy= "
					# 	# print acc

					# 	statistical_matrix[line_num, epoch] = loss

					# 	line_num += 1

					##### small dataset
						# acc = sess.run(accuracy, feed_dict={x: x_input, y: y_input})
						# Calculate batch loss
					loss = sess.run(cross_entropy, feed_dict={x: x_input, y: y_input})

					statistical_matrix[line_num, epoch] = loss

					line_num += 1

					if count % display_step == 0:
						print count,'th chunk.'
					
					count += 1


		print epoch,'th epoch. ----------'

	np.savetxt("statistical_5percent.csv", statistical_matrix, delimiter=",")

	save_path = saver.save(sess, "/home/yigengzhang/workspace/PreProcess/Save_variables/model.ckpt")
	print("Model saved in file: %s" % save_path)


	'''			'''
	''' Testing '''
	'''			'''
	with open(path_data_to_test) as f_test:

		count_test = 0

		# list 1: original labels in total. 1*69. e.g. [208, 1140, 67, 65, ..., 564]
		# list 2: predicted labels in total. 1*69. e.g. [188, 1240, 77, 64, ..., 787] 
		# list 3: correct (true positive) labels in total. 1*69. e.g. [167, 1000, 40, 50, ..., 500] 
		list_1 = np.zeros(69)
		list_2 = np.zeros(69)
		list_3 = np.zeros(69)

		for key,group in itertools.groupby(f_test ,isa_group_separator):

			if not key:

				x_test, y_test, length_of_current_test_sentence = generate_input(group)

				test_sentence_length = 0
				# check sentence length, if padding, cut up; else 60 max

				# test_sentence_length = lambda length_of_current_test_sentence: length_of_current_test_sentence \
				# 						if length_of_current_test_sentence > max_sentence_length else max_sentence_length


				if length_of_current_test_sentence <= max_sentence_length:

					test_sentence_length = length_of_current_test_sentence

				else:

					test_sentence_length = max_sentence_length

				# list 1: Real Label
				y_test_sum = np.sum(y_test, axis=0)
				list_1 = np.sum([list_1, y_test_sum], axis=0)

				# list 2: Predicted Label
				argmaxpredict = sess.run(argmax_pred, feed_dict={x: x_test, y: y_test})
				argmaxpredict_cut = argmaxpredict[:test_sentence_length]

				for element_2 in argmaxpredict_cut:
					list_2[element_2] += 1

				#list 3: True Positive
				correctpredict_ind = sess.run(correct_pred_index, feed_dict={x: x_test, y: y_test})		
				correctpredict_ind_cut = correctpredict_ind[:test_sentence_length]
				# [0,0,1,0,1,0,0,0,1,0...]

				for seq_num in range(test_sentence_length):
					# print seq_num
					if correctpredict_ind_cut[seq_num] == 1:
						list_3[argmaxpredict_cut[seq_num]] += 1

				if count_test % 1000 == 0:
					print 'Testing: '
					print count_test,'th chunk.'

				count_test += 1

		print "list 1:"
		print list_1
		print "list 2:"
		print list_2
		print "list 3:"
		print list_3

		np.savetxt("list_105.csv", list_1, delimiter=",")
		np.savetxt("list_205.csv", list_2, delimiter=",")
		np.savetxt("list_305.csv", list_3, delimiter=",")

		Precision = div_igonre0(list_3,list_2)
		Recall = div_igonre0(list_3,list_1)
		F1 = div_igonre0( np.multiply(2, np.multiply(Precision, Recall)) , np.add(Precision, Recall) )

		print "Precision:"
		print Precision
		print "Recall:"
		print Recall
		print "F1:"
		print F1

		np.savetxt("Precision.csv", Precision, delimiter=",")
		np.savetxt("Recall.csv", Recall, delimiter=",")
		np.savetxt("F1.csv", F1, delimiter=",")

