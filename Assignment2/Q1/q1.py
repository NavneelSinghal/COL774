import sys
import os
import json
import string
import re
import numpy as np
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sklearn
import matplotlib.pyplot as plt
import itertools

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
    scores = np.zeros((len(x_test), numClasses))
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
            scores[len(ans), k] = p
            if p > p_max:
                p_max = p
                k_max = k
        ans.append(k_max + 1)
    return ans, scores

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
    if processing not in ['simple', 'stemming', 'bigram+logfreq', 'bigram', 'logfreq']:
        raise Exception('Invalid processing method')
    isBigram = processing.find('bigram') != -1
    isLogFreq = processing.find('logfreq') != -1
    isStem = isBigram or isLogFreq or (processing.find('stem') != -1)
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
        if not isStem:
            toAdd = x_list
        else:
            for word in x_list:
                if word not in toAvoid:
                    toAdd.append(stem(word))
            if isBigram:
                toAdd = [toAdd[i] + ' ' + toAdd[i + 1] for i in range(len(toAdd) - 1)]
        if isLogFreq:
            x_data.append(logfreq_dict(freq_dict(toAdd)))
        else:
            x_data.append(freq_dict(toAdd))
    return x_data, y_data

def printAccuracy(x_test, y_test, theta, phi, isTest=True):
    print('testing model on', ('test' if isTest else 'training'), 'data...')
    y_output, y_scores = test(x_test, theta, phi)
    print('testing complete')
    confusionMatrix = np.array([[0 for i in range(5)] for j in range(5)])
    for i in range(len(y_output)):
        confusionMatrix[y_output[i] - 1][y_test[i] - 1] += 1
    diagonal = np.sum(np.array([confusionMatrix[i][i] for i in range(5)]))
    total = np.sum(confusionMatrix)
    print(diagonal, '/', total)
    print(diagonal / total)
    print(confusionMatrix)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = sklearn.preprocessing.label_binarize(np.array(y_test), classes=[1, 2, 3, 4, 5])
    n_classes = y_scores.shape[1]
    y_scores_max = np.array([np.amax(y_scores, axis=1)] * n_classes).T
    y_scores = np.exp(y_scores - y_scores_max)
    y_scores /= np.array([np.sum(y_scores, axis=1)] * n_classes).T
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    plt.figure()
    colors = itertools.cycle(['red', 'yellow', 'blue', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i + 1, roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]),
            color='black', linestyle=':', lw=1)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(('test_roc' if isTest else 'train_roc') + '.png')

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
        y, _ = test(x_test, theta, phi)
        write_predictions(sys.argv[3], y)

# submission parameters: 'bigram+logfreq', False
if __name__ == '__main__':
    #main(processing='simple', toPrint=False)
    #main(processing='stemming', toPrint=False)
    #main(processing='logfreq', toPrint=False)
    #main(processing='bigram', toPrint=False)
    main(processing='bigram+logfreq', toPrint=False)
