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

# vocabulary mapping word to word number
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
    # phi(k) = phi[k] / sum(phi[k]) over all k
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

def check_test_random(x_test, y_test, numClasses):
    y_pred = np.random.randint(numClasses, size=len(x_test)) + 1
    correct = 0
    for i in range(len(x_test)):
        if y_pred[i] == y_test[i]:
            correct += 1
    print(correct / len(x_test))

def check_test_majority(x_test, y_test, y_train, numClasses):
    freq = [0 for i in range(numClasses)]
    for i in y_train:
        freq[i - 1] += 1
    cl = freq.index(max(freq)) + 1
    y_pred = np.array([cl for i in range(len(x_test))])
    correct = 0
    for i in range(len(x_test)):
        if y_pred[i] == y_test[i]:
            correct += 1
    print(correct / len(x_test))

def test(x_test, theta, phi):
    global dictionary
    numClasses = len(phi)
    ans = [] # predictions
    scores = np.zeros((len(x_test), numClasses)) # log probabilities
    for x in x_test:
        p_max = -math.inf
        k_max = -1
        for k in range(numClasses):
            theta_k = theta[k]
            p = math.log(phi[k])
            for word in x:
                if word in dictionary: # consider only those words in vocabulary
                    p += math.log(theta_k[dictionary[word]]) * x[word]
            scores[len(ans), k] = p
            if p > p_max:
                p_max = p
                k_max = k
        ans.append(k_max + 1)
    return ans, scores

# clean string by removing all punctuation and splitting
def cleanPunctuation(s):
    return re.sub(r'[^\w\s]', ' ', s.lower()).split()

stemmingDictionary = dict()

# perform cached stemming
def stem(s):
    if s in stemmingDictionary:
        return stemmingDictionary[s]
    else:
        word = stemmer.stem(s)
        stemmingDictionary[s] = word
        return word

# convert list to a dictionary of frequencies
def freq_dict(l):
    d = dict()
    for w in l:
        if w in d:
            d[w] += 1
        else:
            d[w] = 1
    return d

# log frequency calculation
def logfreq_dict(d):
    f = dict()
    for w in d:
        f[w] = math.log2(1 + d[w])
    return f

# parse and process data appropriately
def getData(dataFile, processing='simple', doStem=True):
    if processing not in ['simple', 'stemming', 'bigram+logfreq', 'bigram', 'logfreq']:
        raise Exception('Invalid processing method')
    # whether we need to do stemming, bigrams, log frequencies
    isBigram = processing.find('bigram') != -1
    isLogFreq = processing.find('logfreq') != -1
    isStem = (isBigram or isLogFreq or (processing.find('stem') != -1)) and doStem
    data = []
    # loading json
    with open(dataFile) as data_:
        for line in data_:
            data.append(json.loads(line))
    x_data = []
    y_data = [int(x['stars']) for x in data]
    toAvoid = set(stopwords.words('english'))
    # cleaning and processing
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
    # constructing the confusion matrix
    confusionMatrix = np.array([[0 for i in range(5)] for j in range(5)])
    for i in range(len(y_output)):
        confusionMatrix[y_output[i] - 1][y_test[i] - 1] += 1
    diagonal = np.sum(np.array([confusionMatrix[i][i] for i in range(5)]))
    total = np.sum(confusionMatrix)
    # accuracy-related statistics
    print(diagonal, '/', total)
    print(diagonal / total)
    print(confusionMatrix)
    # ROC-related statistics
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = sklearn.preprocessing.label_binarize(np.array(y_test), classes=[1, 2, 3, 4, 5])
    n_classes = y_scores.shape[1]
    # converting scores to class probabilities
    y_scores_max = np.array([np.amax(y_scores, axis=1)] * n_classes).T
    y_scores = np.exp(y_scores - y_scores_max)
    y_scores /= np.array([np.sum(y_scores, axis=1)] * n_classes).T
    # computing ROC-related stuff
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    from scipy import interp
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])
    # plotting ROC
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
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='violet', linestyle=':', lw=1)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # saving the curve
    plt.savefig(('test_roc' if isTest else 'train_roc') + '.png')

# saving predictions to file
def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

# main function
# params:
#   processing: the type of feature processing we need to do
#   toPrint: whether to print statistics or not
def main(processing='simple', toPrint=True, doStem=True):

    if toPrint: print('processing type:', processing)
    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]

    if toPrint: print('parsing training data')
    x_train, y_train = getData(trainDataFile, processing, doStem)
    if toPrint: print('parsing testing data')
    x_test, y_test = getData(testDataFile, processing, doStem)
    if toPrint: print('training model')
    theta, phi = train(x_train, y_train, 5)

    if toPrint:
        printAccuracy(x_test, y_test, theta, phi, True)
        printAccuracy(x_train, y_train, theta, phi, False)
    else:
        y, _ = test(x_test, theta, phi)
        write_predictions(sys.argv[3], y)

def heuristicMain():
    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]
    x_train, y_train = getData(trainDataFile, 'simple')
    x_test, y_test = getData(testDataFile, 'simple')
    print('random prediction')
    check_test_random(x_test, y_test, 5)
    print('majority prediction')
    check_test_majority(x_test, y_test, y_train, 5)

# submission parameters: 'bigram+logfreq', False
if __name__ == '__main__':
    #heuristicMain()
    #main(processing='simple', toPrint=False)
    #main(processing='stemming', toPrint=False)
    #main(processing='logfreq', toPrint=False)
    #main(processing='bigram', toPrint=False)
    #main(processing='bigram+logfreq', toPrint=False)
    main(processing='bigram+logfreq', toPrint=False, doStem=False)
