import re
import codecs
import json
import time
import numpy as np
from utils import *

def get_vocab(inputFile, outputFile):
    '''
    :param inputFile: sentence file
    :param outputFie: vocabulary file
    '''
    outData = open(outputFile, 'wb')
    wordDict = dict()
    with open(inputFile, 'rb') as inData, open(outputFile,'wb') as outData:
        for line in inData.readlines():
            line = line.strip()
            wordList = re.split('\s+', line)

            for word in wordList:
                if wordDict.get(word) is None:
                    wordDict[word] = 0
                wordDict[word] += 1

        added_word = ['<padding>', '<s>', '</s>', '<unk>']
        for word in added_word:
            print >> outData, word

        sortedWordList = sorted(wordDict.items(), key = lambda d: d[1], reverse = True)
        for item in sortedWordList[:20000]:
            print >> outData, item[0]

def getTokenList(inFile):
    '''
    :param inFile: input file
    ;return idx2token, token2idx
    '''
    i = 0
    idx2token = dict()
    token2idx = dict()
    with open(inFile, 'rb') as inputs:
        for line in inputs.readlines():
            token = line.strip()
            idx2token[i] = token
            token2idx[token] = i
            i += 1

    return idx2token, token2idx

def trainData2idxPairs(inFile, outFile, token2idx):

    with open(inFile, 'rb') as inData, open(outFile, 'wb') as outData:
        sentList = inData.readlines()
        for line in sentList:
            line = line.strip()
            tokenList = re.split('\s+', line)

            # idxList = [str(token2idx[token]) for token in tokenList]
            idxList = list()
            for token in tokenList:
	        if token2idx.get(token) is None:
	            token = '<unk>' 
                idxList.append(str(token2idx[token]))        
  
            first_idxList = [str(token2idx['<s>'])] + idxList
            last_idxList =  idxList + [str(token2idx['</s>'])]

            print >> outData, ' '.join(first_idxList) + '\t' + ' '.join(last_idxList)

def main():
    print "getting vocabulary..."
    t0 = time.time()
    get_vocab(path_train_file, path_vocab)
    t1 = time.time()
    print "Cost: %fs." % (t1-t0)

    print "getting idx2token and token2idx..."
    t0 = time.time()
    idx2token, token2idx = getTokenList(path_vocab)
    t1 = time.time()
    print "Cost: %fs." % (t1-t0)

    print "getting index pairs file..."
    t0 = time.time()
    trainData2idxPairs(path_train_file, path_train, token2idx)
    trainData2idxPairs(path_validation_file, path_validation, token2idx)
    trainData2idxPairs(path_test_file, path_test, token2idx)
    t1 = time.time()
    print "Cost: %fs." % (t1-t0)

    print "saving idx2token and token2idx file..."
    t0 = time.time()
    with open(path_idx2token, 'wb') as mf:
        json.dump(idx2token, mf)
    with open(path_token2idx, 'wb') as mf:
        json.dump(token2idx, mf)
    t1 = time.time()
    print "Cost: %fs." % (t1-t0)

if __name__ == "__main__":

    main()
