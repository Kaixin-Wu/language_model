import theano
import theano.tensor as T 
import numpy as np 
import time
import cPickle

import utils

class GRU_LM(object):
    def __init__(self, vocab_size, hidden_size, lr, batch_size, method="adadelta"):

        self.hidden_size = hidden_size
        self.threshold = utils.get_threshold(hidden_size, hidden_size)
        self.wordEmbedding = theano.shared(name="wordEmbedding", value=utils.init_randn(vocab_size, hidden_size), 
            borrow=True)

        W = self.threshold * np.random.randn(3, hidden_size, hidden_size)
        U = self.threshold * np.random.randn(3, hidden_size, hidden_size)
        b = np.zeros((3, hidden_size))

        self.W = theano.shared(name="W", value=W.astype(theano.config.floatX), borrow=True)
        self.U = theano.shared(name="U", value=U.astype(theano.config.floatX), borrow=True)
        self.b = theano.shared(name="b", value=b.astype(theano.config.floatX), borrow=True)

        self.V = theano.shared(name="V", value=np.zeros((hidden_size, vocab_size), dtype=theano.config.floatX), 
            borrow=True)

        self.lr = lr 
        self.params = [self.wordEmbedding, self.W, self.U, self.b, self.V]
        
        if method == "adadelta":
            self.mean_delta_g2 = [theano.shared(np.zeros_like(param.get_value())) for param in self.params]
            self.mean_delta_x2 = [theano.shared(np.zeros_like(param.get_value())) for param in self.params]

        print 'ok1' 
        index = T.imatrix("index")               # (max_sentence_size, batch_szie)
        x = self.wordEmbedding[index]            # (max_sentence_size, batch_size, hidden_size)
        mask = T.imatrix("mask")                 # (max_sentence_size, batch_size)  mask of input x
        y_expect = T.imatrix("y_expect")         # (max_sentence_size, batch_size)
        print 'ok2'

        h = self.forward(x)
        temp = T.dot(h, self.V)
        y, updates = theano.scan(fn=lambda item: T.nnet.softmax(item),
                 sequences=temp)
        print 'ok6'
        print 'y: ', type(y), y.type

        loss, updates = theano.scan(fn=self.calcuCost,
                                    sequences=[y, y_expect, mask])
        print 'ok7'

        probSum = T.sum(loss)
        totalNum = T.sum(mask)
        cost = probSum / totalNum

        paramsGrads = [T.grad(cost, param) for param in self.params]
        if method == None:
            paramsUpdates = [(param, param + self.lr * g) for param, g in zip(self.params, paramsGrads)]
	
	if method == "adadelta":
            delta_x_updates, delta_g2_updates, delta_x2_updates = self.adaDelta(paramsGrads, decay_rate=0.95, eps=1e-6)

            gradUpdates = [(param, param - delta_x) for param, delta_x in zip(self.params, delta_x_updates)]
            g2Updates = [(oldValue, newValue) for oldValue, newValue in zip(self.mean_delta_g2, delta_g2_updates)]
            x2Updates = [(oldValue, newValue) for oldValue, newValue in zip(self.mean_delta_x2, delta_x2_updates)]

            paramsUpdates = gradUpdates + g2Updates + x2Updates        

	self.train = theano.function(inputs=[index, y_expect, mask],
                                     outputs=[cost, probSum, totalNum],
                                     updates=paramsUpdates)

        self.predict = theano.function(inputs=[index, y_expect, mask],
                                       outputs=[probSum, totalNum])
        

    def forward(self, inputs):
        
        if 3 == inputs.ndim:
            batch_size = inputs.shape[1]
        else:
            batch_size = 1

        print 'ok3' 
        h0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)
        def oneStep(x, h_tm1):
            print 'ok4'
            z = T.nnet.sigmoid(T.dot(x, self.W[0]) + T.dot(h_tm1, self.U[0]) + self.b[0])  # (batch_size, hidden_size)
            r = T.nnet.sigmoid(T.dot(x, self.W[1]) + T.dot(h_tm1, self.U[1]) + self.b[1])  # (batch_size, hidden_size)
            h_ = T.tanh(T.dot(x, self.W[2]) + T.dot(r*h_tm1, self.U[2]) + self.b[2])       # (batch_size, hidden_size)
            h_t = (1. - z)*h_tm1 + z*h_
            print 'ok5'

            return h_t

        h, updates = theano.scan(fn=oneStep,
                                sequences=inputs,
                                outputs_info=h0)
        print 'ok6'
        print 'h: ', type(h), h.type

        return h
    
    def calcuCost(self, y_t, y_expect_t, msk):
        return T.sum((T.log(y_t)[T.arange(y_t.shape[0]), y_expect_t]) * msk)

    def adaDelta(self, paramsGrads, decay_rate=0.95, eps=1e-6):

        delta_x_updates = list()
        delta_g2_updates = list()
        delta_x2_updates = list()
        for g, delta_g2, delta_x2 in zip(paramsGrads, self.mean_delta_g2, self.mean_delta_x2):
            delta_g2 = decay_rate * delta_g2 + (1. - decay_rate) *  (g**2)
            delta_g2_updates.append(delta_g2)

            delta_x = -T.sqrt((delta_x2 + eps) / (delta_g2 + eps)) * g
            delta_x_updates.append(delta_x)

            delta_x2 = decay_rate * delta_x2 + (1. - decay_rate) *  (delta_x**2)
            delta_x2_updates.append(delta_x2)

        return delta_x_updates, delta_g2_updates, delta_x2_updates
