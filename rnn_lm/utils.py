import re
import theano
import numpy as np
import json
import random

path_train_file = "../rnn_lm/data/brown.corpus.train"
path_validation_file = "../rnn_lm/data/brown.corpus.validation"
path_test_file = "../rnn_lm/data/brown.corpus.test"
path_vocab = "../rnn_lm/data/vocab.EN"

path_idx2token = "../rnn_lm/data/idx2token.json"
path_token2idx = "../rnn_lm/data/token2idx.json"
path_train = "../rnn_lm/data/train.txt"
path_validation = "../rnn_lm/data/validation.txt"
path_test = "../rnn_lm/data/test.txt"
path_pkl = "../rnn_lm/data/model/model.pkl"

def shuffleData(x, y):

    inputs = list()
    for x_item, y_item in zip(x, y):
        sample = (x_item, y_item)
        inputs.append(sample)
    random.shuffle(inputs)

    x, y = list(), list()
    for item in inputs:
        x.append(item[0])
        y.append(item[1])

    return x, y

def get_threshold(n_in, n_out):
    return 1. * np.sqrt(6. / (n_in + n_out))

def init_randn(n_in, n_out):
    threshold = 1. * np.sqrt(6. / (n_in + n_out))
    W = np.asarray(threshold * np.random.randn(n_in, n_out), dtype=theano.config.floatX)
    
    return W

def load_train_data(data_path, num):
    x, y = [], []
    with open(data_path, 'rb') as mf:
        lines = mf.readlines()
        for line in lines[:num]:
            x_item, y_item = line.strip().split('\t')
            x += [int(item) for item in x_item.split(' ')], # append
            y += [int(item) for item in y_item.split(' ')], # append

    return x, y

def get_array(inputList, reversed_flag=False):
    '''
    :param inputList:
    '''
    output = list()
    max_size = max([len(item) for item in inputList])
    for item in inputList:
        temp = item + [0]*(max_size - len(item))
        if reversed_flag:
            temp = temp[::-1]
        
        output.append(np.asarray(temp))

    return np.asarray(output, dtype="int32").T

def get_mask(data):
    '''
    get mask of data
    '''
    mask = (np.not_equal(data, 0)).astype("int32")
    return mask
