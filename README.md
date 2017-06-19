# DL4SRL
Semantic Role Labeling Using Deep Learning Method

File 'feed_to_network_new.py' is the main pipeline for training. It reads training data and feed the neural network with one sentence each time. This file needs to read a word2vec model(Chinese) to convert words to vectors. And it also need a 'label dictionary' to map labels to numbers(of classification).

File 'labels_dictionary.txt' is for 'feed_to_network.py' to read.

Please find dataset in folder 'Data'.

File folder 'Preprocess' contains some files for raw data preprocessing, can just simply ignore it.
