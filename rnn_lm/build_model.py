# -*- coding: UTF-8 -*
import theano
import theano.tensor as T 
import numpy as np 
import cPickle
import time
import timeit
import codecs

from utils import *
from rnn_lm_v2 import *

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

sys.setrecursionlimit(150000000)  # cPickle

outputResult = codecs.open("../rnn_lm/data/result.txt", "w")
testResult = codecs.open("../rnn_lm/data/testResult.txt", "w") 

def build_model(vocab_size=20004, hidden_size=512, batch_size=64, epochs=100, corpus_num=53341, lr=0.1, retrain=False):
    # 57341
    model = None

    if not retrain:
        try:
            f = open(path_pkl)
            model = cPickle.load(f)
            return model
        except Exception:
            print "Model does not pre-exist..."

    print "Start training a new model..."

    # (corpus_num, sent_size)
    x, y = load_train_data(path_train, corpus_num)

    num_batchs = corpus_num // batch_size
    max_epochs = num_batchs * epochs
    model = GRU_LM(vocab_size, hidden_size, lr, batch_size)

    t0 = time.time()
    batch_idx = 0

    probList = list()
    numList = list()
    for epoch in xrange(max_epochs):
        #(batch_size, sent_size)
        input_x = x[batch_idx*batch_size:(batch_idx+1)*batch_size]
        input_y = y[batch_idx*batch_size:(batch_idx+1)*batch_size]

        #(max_sent_size, batch_size)
        input_x = get_array(input_x)
        input_y = get_array(input_y)
        mask = get_mask(input_x)
        
        loss, probSum, totalNum = model.train(input_x, input_y, mask)
        probList.append(probSum)
        numList.append(totalNum)

        # if (epoch+1) % 25 == 0:
        print "in epoch %d/%d..." % (epoch+1, max_epochs) + "    loss:    " + str(loss) + "\t" + str((epoch+1) / num_batchs)
        if (epoch+1) % num_batchs == 0:
            t1 = time.time()
            # print probList
            # print numList
            Ppl = np.exp(-1.0 * np.sum(probList) / np.sum(numList))
            print "Cost: %.2fs.\tPpl: %.2f\tin Epoch%d" % (t1-t0, Ppl, (epoch+1)/num_batchs)
            print >> outputResult, "Loss: %.2f\tPpl: %.2f\tin Epoch%d" % (loss, Ppl, (epoch+1)/num_batchs)
            probList = list()
            numList = list()
	    
            print >> testResult, "In Epoch%d" % ((epoch+1)/num_batchs)
            print >> testResult, "In Validation Set\t",
            testPpl(model, path_validation)
            print >> testResult, "In Test Set\t",
            testPpl(model, path_test)
        
            num = (epoch+1) / num_batchs
            with open(path_pkl+str(num), "wb") as mf:
                cPickle.dump(model, mf) 
            
            x, y = shuffleData(x, y)

        batch_idx = (batch_idx+1) % num_batchs
    
    with open(path_pkl, "wb") as mf:
        cPickle.dump(model, mf)
        
    return model

def testPpl(model, inFile, corpus_num=2000):
    
    probList = list()
    numList = list()
    x, y = load_train_data(inFile, corpus_num)
    for input_x, input_y in zip(x, y):
        input_x = np.asarray(input_x, dtype="int32")
        input_y = np.asarray(input_y, dtype="int32")
       
        # print input_x
        # print input_y

        input_x = input_x.reshape((input_x.shape[0], 1))
        input_y = input_y.reshape((input_y.shape[0], 1))
        # print input_x
        # print input_y

        mask = get_mask(input_x)
        probSum, totalNum = model.predict(input_x, input_y, mask)
        probList.append(probSum)
        numList.append(totalNum)

    Ppl = np.exp(-1.0 * np.sum(probList) / np.sum(numList))
    print >> testResult, "Ppl: %.2f" % Ppl
    # print "Ppl: %.2f" % Ppl


if __name__ == "__main__":
    print "Initializing RNN_LM Model..."
    time_start = timeit.default_timer()

    model = build_model(retrain=True)
    # print "in validation: "
    # testPpl(model, path_validation)
    # print "in test"
    # testPpl(model, path_test)

    time_end = timeit.default_timer()
    print "Done initializing RNN_LM Model...Time taken:   ", (time_end-time_start)
