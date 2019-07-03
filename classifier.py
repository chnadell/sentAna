import tensorflow as tf
import numpy as np
import nltk
import imdb
import time

import os
import sys
import pickle

from sklearn.metrics import confusion_matrix

num_classes = 2  # types of sentence
embedding_size = 300  # dimensions of embedded vectors
num_words = 20000  # max number of words to be tokenized in the corpus
sequence_length = 70  # max length of a sequence. Longer are truncated to this, shorter are padded to this

gru_layer_1 = 16
gru_layer_2 = 8
gru_layer_3 = 4

# training hyperparams
learning_rate = .001
num_epochs = 5
batch_size = 32


# import data


# for splitting data into sentences via NLTK library function, then cut out sentences that are too short, which may be a
# result of improper data parsing.
def file_tokenize(path):
    with open(path, encoding='UTF8', errors='ignore') as f:
        dataIn = f.read().replace('\n', ' ')
    data_tokenized = nltk.sent_tokenize(dataIn)
    data_tokenizedClean = []
    for string in data_tokenized:
        if len(string) > 10:
            data_tokenizedClean.append(string)
        return data_tokenizedClean


# return dataset object of the training data
# tokenizes input sentences
def read_data(data_path):
    print("importing data")

    print('tokenizing the data')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)

    # save the tokenizer
    with open('tokenizer' + str(time.strftime("%Y%m%d-%H%M%S"))) as tokenizerFile:
        pickle.dump(tokenizer, tokenizerFile)


# train the network on some dataset

def make_gru_network(gru_sizes, dense_sizes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_size,
                                        num_words=num_words, sequence_length=sequence_length,
                                        trainable=True,name='embedding_layer'))
    for cnt, gru_size in enumerate(gru_sizes):
        model.add(tf.keras.layers.GRU(units=gru_size, name='GRU_' + str(cnt))
                  )
    for cnt, dense_size in enumerate(dense_sizes):
        model.add(tf.keras.layers.Dense(units=dense_size, name='dense_' + str(cnt)))


def train(dataset, keras_model, lr=learning_rate, batch_size=batch_size, epochs=num_epochs):
    # define optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.keras.optimizers.Adam(lr=lr)

    # compile model with a given loss and optimizer, etc
    with tf.name_scope('compiler'):
        keras_model.compile(optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    keras_model.summary()

    # fit the model
    with tf.name_scope('fit'):
        keras_model.fit(dataset.make_one_shot_iterator(),
                        epochs=epochs,
                        batch_size=batch_size)
