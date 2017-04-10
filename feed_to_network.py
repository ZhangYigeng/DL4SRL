# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import numpy as np
import gensim
import itertools


import time

import tensorflow as tf

# directories:
path_word2vec_model = '/home/yigengzhang/workspace/Word2Vec/wiki.zh.text.new.model'
path_label_dictionary = '/home/yigengzhang/workspace/PreProcess/labels_dictionary.txt'
path_data_to_process = '/home/yigengzhang/workspace/PreProcess/SplitedSentenceSample'



model = gensim.models.Word2Vec.load(path_word2vec_model)
vec_length = 400


def label_dic():
	dictionary = np.loadtxt(path_label_dictionary, delimiter=' ', dtype = np.str)
	label_dictionary = dict(zip(dictionary[:,0], dictionary[:,1]))
	return label_dictionary




def isa_group_separator(line):
	return line == '\n'

def look_up(word): 		

	word = unicode(word, 'utf-8')
	# print word
	# print model[word]
	try:
		return model[word]

	except KeyError:
		no_vector = np.array(range(vec_length)).reshape(1,vec_length)
		no_vector.fill(0)
		print "not in vocabulary"
		return no_vector

# construct vector for each word in sentence (argu,pred,ctx_p_1,2,3)
def construct_vector(sequence):
	vector = []
	# look up each word
	for i in range(5):
		vector.append(look_up(sequence[i]))

	# region_mark = np.array(range(vec_length)).reshape(1,vec_length)
	# region_mark.fill(float(sequence[5]))
	# vector.append(region_mark)
	represent_as_vector = np.array(vector)
	print represent_as_vector.size
	# represent_as_vector = np.ravel(represent_as_vector)
	# represent_as_vector = np.append(represent_as_vector, (np.array(sequence[5], dtype = np.float32)))

	return represent_as_vector



lstm_size = 400
batch_size = 5
data_type = tf.float32
count = 0
label_dictionary = label_dic()

# pipeline
# input = tf.placeholder(tf.float32)
predict_label = tf.placeholder(tf.float32)
real_label = tf.placeholder(tf.float32)
lstm_output = tf.placeholder(tf.float32)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size,forget_bias=0.0, state_is_tuple=False)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 8)
initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
# initial_state = state = tf.zeros([batch_size, lstm_cell.state_size])

softmax_w = tf.Variable(tf.random_normal([400, 69], stddev=0.35), name="softmax_w")
softmax_b = tf.Variable(tf.random_normal([69], stddev=0.35), name="softmax_b")

# softmax_w = tf.get_variable("softmax_w", [400, 69], dtype=data_type)
# softmax_b = tf.get_variable("softmax_b", [69], dtype=data_type)

predict_label = tf.matmul(lstm_output, softmax_w) + softmax_b
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(real_label * tf.log(test_label), reduction_indices=[1]))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = real_label,logits = predict_label)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# sess = tf.Session()
init = tf.global_variables_initializer()

with tf.Session() as sess:
	
	sess.run(init)
	sess.run(state)

	print "initialized!!!"

	with open(path_data_to_process) as f:

		for key,group in itertools.groupby(f,isa_group_separator):
			# print(key,list(group))  # uncomment to see what itertools.groupby does.
			if not key:
				# each sentence
				for sentence in group:
					sequence = np.asarray(sentence.strip().split(" "))
					# vectorization
					long_vector = construct_vector(sequence)
					# print long_vector.size

					# label for each sentence
					label = int(label_dictionary[sequence[6]])
					label_vector = np.zeros(69)
					label_vector[label-1] = 1.0
					# print label_vector
					# sess.run(state)

					output, state = stacked_lstm(tf.convert_to_tensor(long_vector), state)
					print output
					# sess = tf.Session()

					network_output, network_state = sess.run([output, state])
					sess.run(cross_entropy,{real_label:label_vector, predict_label:network_output})
					sess.run(train, {real_label:label_vector, lstm_output:network_output})


				

			count = count + 1
			print count,'th sentence'




