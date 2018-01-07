import re
import random
import codecs
import random

def randomDistribute(inFile, trainFile, validationFile, testFile, num=2000):
    
    inData = codecs.open(inFile, 'r').readlines()
    trainOutput = codecs.open(trainFile, 'w')
    validationOutput = codecs.open(validationFile, 'w')
    testOutput = codecs.open(testFile, 'w')
	
    random.shuffle(inData)
    trainData = inData[num:-num]
    validationData = inData[:num]
    testData = inData[-num:]
    
    print "train data: ", len(trainData)
    print "validation data: ", len(validationData)
    print "test data: ", len(testData)
    print "total: ", len(inData)

    print >> trainOutput, ''.join(trainData),
    print >> validationOutput, ''.join(validationData),
    print >> testOutput, ''.join(testData),

if __name__ == "__main__":
    
    inFile = "../rnn_lm/data/brown.corpus"
    trainFile = inFile + ".train"
    validationFile =  inFile + ".validation"
    testFile = inFile + ".test"

    randomDistribute(inFile, trainFile, validationFile, testFile)




