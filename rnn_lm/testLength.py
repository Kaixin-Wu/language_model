import re

inFile = "../rnn_lm/data/brown.corpus.train"
inData = open(inFile, 'r')

maxLen = 0
flag = 0
i = 0
for line in inData.readlines():
    line = line.strip().split(" ")
    
    if len(line) > maxLen:
        maxLen = len(line)
        flag = i
    i+=1

print maxLen, flag
