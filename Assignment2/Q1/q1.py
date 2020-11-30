import sys
import os
import json
import string
import re
import numpy as np
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()

dictionary = dict()
dictionarySize = 0

# returns parameters theta, phi
def train(x_train, y_train, numClasses):
    global dictionary, dictionarySize
    # populating dictionary
    dictionary = dict()
    for x in x_train:
        for word in x:
            if word not in dictionary:
                dictionary[word] = dictionarySize
                dictionarySize += 1
    # theta(j = l | k) = theta_num[k][l] / sum(theta_num[k][l]) over all l
    theta_num = np.ones((numClasses, dictionarySize))
    phi = np.zeros(numClasses)
    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i] - 1
        # -1 since 1-indexed
        phi[y_i] += 1
        for w in x_i:
            theta_num[y_i][dictionary[w]] += x_i[w]
    for i in range(numClasses):
        theta_num[i] /= np.sum(theta_num[i])
    phi /= np.sum(phi)
    return (theta_num, phi)

def test(x_test, theta, phi):
    global dictionary
    numClasses = len(phi)
    ans = []
    for x in x_test:
        p_max = -1e9
        k_max = -1
        for k in range(numClasses):
            phi_k = phi[k]
            theta_k = theta[k]
            p = math.log(phi_k)
            for word in x:
                if word in dictionary:
                    p += math.log(theta_k[dictionary[word]]) * x[word]
            if p > p_max:
                p_max = p
                k_max = k
        ans.append(k_max + 1)
    return ans

def cleanPunctuation(s):
    return re.sub(r'[^\w\s]', ' ', s.lower()).split()

stemmingDictionary = dict()

def stem(s):
    if s in stemmingDictionary:
        return stemmingDictionary[s]
    else:
        word = stemmer.stem(s)
        stemmingDictionary[s] = word
        return word

def freq_dict(l):
    d = dict()
    for w in l:
        if w in d:
            d[w] += 1
        else:
            d[w] = 1
    return d

def logfreq_dict(d):
    f = dict()
    for w in d:
        f[w] = math.log2(1 + d[w])
    return f

def getData(dataFile, processing='simple'):
    if processing not in ['simple', 'stemming', 'bigram+logfreq']:
        raise Exception('Invalid processing method')
    data = []
    with open(dataFile) as data_:
        for line in data_:
            data.append(json.loads(line))
    x_data = []
    y_data = [int(x['stars']) for x in data]
    toAvoid = set(stopwords.words('english'))
    for x in data:
        x_list = cleanPunctuation(x['text'])
        toAdd = []
        if processing == 'simple':
            toAdd = x_list
        else:
            for word in x_list:
                if word not in toAvoid:
                    toAdd.append(stem(word))
            if processing == 'bigram+logfreq':
                toAdd = [toAdd[i] + ' ' + toAdd[i + 1] for i in range(len(toAdd) - 1)]
        if processing == 'bigram+logfreq':
            x_data.append(logfreq_dict(freq_dict(toAdd)))
        else:
            x_data.append(freq_dict(toAdd))
    return x_data, y_data

def printAccuracy(x_test, y_test, theta, phi, isTest=True):
    print('testing model on', ('test' if isTest else 'training'), 'data...')
    y_output = test(x_test, theta, phi)
    print('testing complete')
    confusionMatrix = np.array([[0 for i in range(5)] for j in range(5)])
    for i in range(len(y_output)):
        confusionMatrix[y_output[i] - 1][y_test[i] - 1] += 1
    diagonal = np.sum(np.array([confusionMatrix[i][i] for i in range(5)]))
    total = np.sum(confusionMatrix)
    print(diagonal, '/', total)
    print(diagonal / total)
    print(confusionMatrix)

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def main(processing='simple', toPrint=True):

    if toPrint: print('processing type:', processing)
    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]

    if toPrint: print('parsing training data')
    x_train, y_train = getData(trainDataFile, processing)
    if toPrint: print('parsing testing data')
    x_test, y_test = getData(testDataFile, processing)
    if toPrint: print('training model')
    theta, phi = train(x_train, y_train, 5)

    if toPrint:
        printAccuracy(x_test, y_test, theta, phi, True)
        printAccuracy(x_train, y_train, theta, phi, False)
    else:
        write_predictions(sys.argv[3], test(x_test, theta, phi))

# submission parameters: 'bigram+logfreq', False
if __name__ == '__main__':
    main(processing='bigram+logfreq', toPrint=False)
