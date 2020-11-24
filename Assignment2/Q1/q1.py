import sys
import os
import json
import string
import re
import numpy as np
import pickle
import math
from nltk.stem import PorterStemmer
#from nltk.stem import SnowballStemmer
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stemmer = PorterStemmer()
#stemmer = SnowballStemmer('english')
#lemmatizer = WordNetLemmatizer()

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
    #print(dictionary)
    #x_train = [[dictionary[word] for word in x] for x in x_train]
    # theta(j = l | k) = theta_num[k][l] / theta_den[k]
    theta_num = np.ones((numClasses, dictionarySize))
    theta_den = dictionarySize * np.ones(numClasses)
    phi = np.zeros(numClasses)
    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i] - 1
        # -1 since 1-indexed
        theta_den[y_i] += len(x_i)
        phi[y_i] += 1
        for w in x_i:
            theta_num[y_i][dictionary[w]] += 1
    for i in range(numClasses):
        theta_num[i] /= theta_den[i]
    phi /= np.sum(phi)
    #print(theta_num[:, 0:10], phi)
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
                    p += math.log(theta_k[dictionary[word]])
            if p > p_max:
                p_max = p
                k_max = k
        ans.append(k_max + 1)
    return ans

def cleanPunctuation(s):
    return re.sub(r'[^\w\s]', ' ', s.lower()).split()

#def retrieveData(trainDataFile, testDataFile):
#    trainData, testData = [], []
#    x_train, y_train, x_test, y_test = None, None, None, None
#    if os.path.isfile(os.path.join('.', 'clean.pickle')):
#        x_train, y_train, x_test, y_test = pickle.load(open('clean.pickle', 'rb'))
#    else:
#        with open(trainDataFile) as train_:
#            for line in train_:
#                trainData.append(json.loads(line))
#        with open(testDataFile) as test_:
#            for line in test_:
#                testData.append(json.loads(line))
#        x_train = [cleanPunctuation(x['text']) for x in trainData]
#        y_train = [int(x['stars']) for x in trainData]
#        x_test = [cleanPunctuation(x['text']) for x in testData]
#        y_test = [int(x['stars']) for x in testData]
#        pickle.dump((x_train, y_train, x_test, y_test), open('clean.pickle', 'wb'))
#    return x_train, y_train, x_test, y_test

def freshRetrieveData(trainDataFile, testDataFile):
    trainData, testData = [], []
    with open(trainDataFile) as train_:
        for line in train_:
            trainData.append(json.loads(line))
    with open(testDataFile) as test_:
        for line in test_:
            testData.append(json.loads(line))
    x_train = [cleanPunctuation(x['text']) for x in trainData]
    y_train = [int(x['stars']) for x in trainData]
    x_test = [cleanPunctuation(x['text']) for x in testData]
    y_test = [int(x['stars']) for x in testData]
    return x_train, y_train, x_test, y_test

stemmingDictionary = dict()

def cleanUp(s):
    if s in stemmingDictionary:
        return stemmingDictionary[s]
    else:
        word = stemmer.stem(s)
        #word = lemmatizer.lemmatize(s)
        stemmingDictionary[s] = word
        return word

def freshRetrieveProcessedData(trainDataFile, testDataFile):
    trainData, testData = [], []
    with open(trainDataFile) as train_:
        for line in train_:
            trainData.append(json.loads(line))
    with open(testDataFile) as test_:
        for line in test_:
            testData.append(json.loads(line))
    y_train = [int(x['stars']) for x in trainData]
    y_test = [int(x['stars']) for x in testData]
    x_train, x_test = [], []
    toAvoid = set(stopwords.words('english'))
    for x in trainData:
        x_list = cleanPunctuation(x['text'])
        toAdd = []
        for word in x_list:
            if word not in toAvoid:
                toAdd.append(cleanUp(word))
        x_train.append([toAdd[i] + ' ' + toAdd[i + 1] for i in range(len(toAdd) - 1)])
    for x in testData:
        x_list = cleanPunctuation(x['text'])
        toAdd = []
        for word in x_list:
            if word not in toAvoid:
                toAdd.append(cleanUp(word))
        x_test.append([toAdd[i] + ' ' + toAdd[i + 1] for i in range(len(toAdd) - 1)])
    return x_train, y_train, x_test, y_test

#def getParameters(x_train, y_train):
#    theta, phi = None, None
#    if os.path.isfile(os.path.join('.', 'parameters.pickle')):
#        theta, phi = pickle.load(open('parameters.pickle', 'rb'))
#    else:
#        theta, phi = train(x_train, y_train, 5)
#        pickle.dump((theta, phi), open('parameters.pickle', 'wb'))
#    return theta, phi

def freshGetParameters(x_train, y_train):
    return train(x_train, y_train, 5)

def printAccuracy(x_test, y_test, theta, phi, isTest=True):
    print('testing model on', ('test' if isTest else 'training'), 'data...')
    y_output = test(x_test, theta, phi)
    print('testing complete')
    misses = 0
    confusionMatrix = np.array([[0 for i in range(5)] for j in range(5)])
    for i in range(len(y_output)):
        if y_output[i] != y_test[i]:
            misses += 1
        confusionMatrix[y_output[i] - 1][y_test[i] - 1] += 1
    print(1 - (misses / len(y_output)))
    diagonal = np.sum(np.array([confusionMatrix[i][i] for i in range(5)]))
    total = np.sum(confusionMatrix)
    print(diagonal, '/', total)
    print(diagonal / total)
    print(confusionMatrix)

def main():

    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]
    outputFile = sys.argv[3]

    #x_train, y_train, x_test, y_test = freshRetrieveData(trainDataFile, testDataFile)
    x_train, y_train, x_test, y_test = freshRetrieveProcessedData(trainDataFile, testDataFile)
    #x_train, y_train, x_test, y_test = retrieveData(trainDataFile, testDataFile)
    theta, phi = freshGetParameters(x_train, y_train)
    #(theta, phi) = getParameters(x_train, y_train)

    printAccuracy(x_test, y_test, theta, phi, True)
    printAccuracy(x_train, y_train, theta, phi, False)

if __name__ == '__main__':
    main()
