# DL4SRL
Semantic Role Labeling Using Deep Learning Method

File 'feed_to_network.py' is the main pipeline for training. It reads training data and feed the neural network with one sentence each time. This file needs to read a word2vec model(Chinese) to convert words to vectors. And it also need a 'label dictionary' to map labels to numbers(of classification).
### there are still bugs have not been solved yet.

File 'labels_dictionary.txt' is for 'feed_to_network.py' to read.

File 'training_dataset_SAMPLE' is a tiny sample of the training data.

File folder 'Preprocess' contains some files for raw data preprocessing, can just simply ignore it.
