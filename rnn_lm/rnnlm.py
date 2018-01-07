import theano
import theano.tensor as T 
import numpy as np 
import time
import cPickle

import utils

class GRU(object):
    def __init__(self, hidden_size):
        '''
        :param hidden_size: the size of hidden layer
        '''
        self.hidden_size = hidden_size
        self.threshold = utils.get_threshold(hidden_size, hidden_size)

        # W = np.random.randn(-np.sqrt(3./hidden_size), np.sqrt(3./hidden_size), (3, hidden_size, hidden_size))
        # U = np.random.randn(-np.sqrt(3./hidden_size), np.sqrt(3./hidden_size), (3, hidden_size, hidden_size))       
        # b = np.zeros((3, hidden_size))
        W = self.threshold * np.random.randn(3, hidden_size, hidden_size)
        U = self.threshold * np.random.randn(3, hidden_size, hidden_size)
        b = np.zeros((3, hidden_size))

        self.W = theano.shared(name="W", value=W.astype(theano.config.floatX), borrow=True)
        self.U = theano.shared(name="U", value=U.astype(theano.config.floatX), borrow=True)
        self.b = theano.shared(name="b", value=b.astype(theano.config.floatX), borrow=True)

        self.params = [self.W, self.U, self.b]

    def forward(self, inputs, mask, h0=None):
        '''
        :param inputs: #(max_sentence_size, batch_size, hidden_size)
        :param mask: the mask of inputs
        '''
        if 3 == inputs.ndim:  # dimension, represent batch_size > 1
            batch_size = inputs.shape[1]
        else:
            batch_size = 1

        if None == h0:
            h0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)

        def oneStep(x, m, h_tm1):
            z = T.nnet.sigmoid(T.dot(x, self.W[0]) + T.dot(h_tm1, self.U[0]) + self.b[0])  # (batch_size, hidden_size)
            r = T.nnet.sigmoid(T.dot(x, self.W[1]) + T.dot(h_tm1, self.U[1]) + self.b[1])  # (batch_size, hidden_size)
            h_ = T.tanh(T.dot(x, self.W[2]) + T.dot(r*h_tm1, self.U[2]) + self.b[2])       # (batch_size, hidden_size)
            h_t = (T.ones_like(z, dtype=theano.config.floatX) - z)*h_tm1 + z*h_                                        # (batch_size, hidden_size)

            # if current position is 0 in mask, the C is same as C
            # m[:, None], dimension is extend.
            h_t = m[:, None]*h_t + (1.0 - m)[:, None]*h_tm1
            h_t = T.cast(h_t, theano.config.floatX)

            return h_t

        hs, updates = theano.scan(
            fn=oneStep,
            sequences=[inputs, mask],
            outputs_info=h0)

        return hs        # (max_sent_size, batch_size, hidden_size)

class GRU_LM(object):
    
    def forward(self, inputs, mask):
        
        if 3 = inputs.ndim:
            batch_size = inputs.shape[1]
        else:
            batch_size = 1

        h0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)
        def oneStep(x, h_tm1):
            z = T.nnet.sigmoid(T.dot(x, self.W[0]) + T.dot(h_tm1, self.U[0]) + self.b[0])  # (batch_size, hidden_size)
            r = T.nnet.sigmoid(T.dot(x, self.W[1]) + T.dot(h_tm1, self.U[1]) + self.b[1])  # (batch_size, hidden_size)
            h_ = T.tanh(T.dot(x, self.W[2]) + T.dot(r*h_tm1, self.U[2]) + self.b[2])       # (batch_size, hidden_size)
            h_t = (1. - z)*h_tm1 + z*h_

            # h_t = m[:, None]*h_t + (1.0 - m)[:, None]*h_tm1
            # h_t = T.cast(h_t, theano.config.floatX)

            return h_t

        h, updates = theano.scan(fn=oneStep,
                                sequences=x,
                                outputs_info=h0)

        return h




    def __init__(self, vocab_size, hidden_size, lr, batch_size):

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
        self.params = [self., self.W, self.U, self.b, self.V]

        print 'ok1' 
        index = T.imatrix()               # (max_sentence_size, batch_szie)
        x = self.wordEmbedding[index]     # (max_sentence_size, batch_size, hidden_size)
        mask = T.imatrix("mask")          # (max_sentence_size, batch_size)  mask of input x
        y_expect = T.imatrix()            # (max_sentence_size, batch)

        h0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, hidden_size)
        def oneStep(x, h_tm1):
            print 'ok3'
            z = T.nnet.sigmoid(T.dot(x, self.W[0]) + T.dot(h_tm1, self.U[0]) + self.b[0])  # (batch_size, hidden_size)
            r = T.nnet.sigmoid(T.dot(x, self.W[1]) + T.dot(h_tm1, self.U[1]) + self.b[1])  # (batch_size, hidden_size)
            h_ = T.tanh(T.dot(x, self.W[2]) + T.dot(r*h_tm1, self.U[2]) + self.b[2])       # (batch_size, hidden_size)
            h_t = (1. - z)*h_tm1 + z*h_
            print 'ok4'                                        # (batch_size, hidden_size)

            # h_t = m[:, None]*h_t + (1.0 - m)[:, None]*h_tm1
            # h_t = T.cast(h_t, theano.config.floatX)

            return h_t

        print 'ok2'
        h, updates = theano.scan(fn=oneStep,
                                sequences=x,
                                outputs_info=h0)

        print type(h), h.type 
        temp = T.dot(h, self.V)
        y, updates = theano.scan(fn=lambda x: T.nnet.softmax(x),
                 sequences=temp)
        print type(y), y.type
        print "ok5"
        def calcuCost(y_t, y_expect_t, msk):
            return T.sum((T.log(y_t)[T.arange(y_t.shape[0]), y_expect_t]) * msk)

        loss, updates = theano.scan(fn=calcuCost,
                                    sequences=[y, y_expect, mask])
        print "ok6"
        probSum = T.sum(loss)
        totalNum = T.sum(mask)
        # cost = T.sum(loss) / T.sum(mask)
        cost = probSum / totalNum

        paramsGrads = [T.grad(cost, param) for param in self.params]
        paramsUpdates = [(param, param + self.lr * g) for param, g in zip(self.params, paramsGrads)]
        self.train = theano.function(inputs=[index, y_expect, mask],
                                     outputs=[cost, probSum, totalNum],
                                     updates=paramsUpdates)


class RNN_LM(object):
    def __init__(self, vocab_size, hidden_size, lr, batch_size):

        self.wordEmbedding = theano.shared(name="wordEmbedding", value=utils.init_randn(vocab_size, hidden_size), 
            borrow=True)
        self.U = theano.shared(name="U", value=utils.init_randn(hidden_size, hidden_size), borrow=True) 
        self.W = theano.shared(name="W", value=utils.init_randn(hidden_size, hidden_size), borrow=True)
        self.V = theano.shared(name="V", value=np.zeros((hidden_size, vocab_size), dtype=theano.config.floatX), 
            borrow=True)

        self.s0 = theano.shared(name="s0", value=np.zeros((batch_size, hidden_size), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(name="b", value=np.zeros(hidden_size, dtype=theano.config.floatX), borrow=True)

        self.lr = lr
        self.params = [self.wordEmbedding, self.U, self.W, self.V, self.b]
        
        print 'ok1' 
        index = T.imatrix()               # (max_sentence_size, batch_szie)
        x = self.wordEmbedding[index]     # (max_sentence_size, batch_size, hidden_size)
        mask = T.imatrix("mask")          # (max_sentence_size, batch_size)  mask of input x
        y_expect = T.imatrix()            # (max_sentence_size, batch)

        def oneStep(x_t, s_tm1):
            print 'ok3'
            s_t = T.nnet.sigmoid(T.dot(x_t, self.U) + T.dot(s_tm1, self.W) + self.b)  # (batch_size, hidden_size)
            print "ok4"
            # y_t = T.nnet.softmax(T.dot(s_t, self.V))                                  # (batch_size, vocab_size)

            return s_t
        
        print 'ok2'
        # y(max_sentence_size, batch_size, vocab_size)
        s, updates = theano.scan(fn=oneStep,
                                 sequences=x,
                                 outputs_info=[self.s0])

        print type(s), s.type 
        temp = T.dot(s, self.V)
        y, updates = theano.scan(fn=lambda x: T.nnet.softmax(x),
				 sequences=temp)
        print type(y), y.type
        print "ok5"
        def calcuCost(y_t, y_expect_t, msk):
            return T.sum((T.log(y_t)[T.arange(y_t.shape[0]), y_expect_t]) * msk)

        loss, updates = theano.scan(fn=calcuCost,
                                    sequences=[y, y_expect, mask])
        print "ok6"
        probSum = T.sum(loss)
        totalNum = T.sum(mask)
        # cost = T.sum(loss) / T.sum(mask)
        cost = probSum / totalNum

        paramsGrads = [T.grad(cost, param) for param in self.params]
        paramsUpdates = [(param, param + self.lr * g) for param, g in zip(self.params, paramsGrads)]
        self.train = theano.function(inputs=[index, y_expect, mask],
                                     outputs=[cost, probSum, totalNum],
                                     updates=paramsUpdates)

