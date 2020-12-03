import sys
import os
import sklearn
import json
import numpy as np
import time

def getXY(dataFile):
    data = []
    with open(dataFile) as data_:
        for line in data_:
            data.append(json.loads(line))
    x_data = [x['text'] for x in data]
    y_data = [int(x['stars']) for x in data]
    return x_data, y_data

def transformData(x_train, y_train, x_test, y_test):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    cv = CountVectorizer()
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)
    tfidf = TfidfTransformer()
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test

def NaiveBayes(x_train, y_train, x_test, y_test, toPrint):
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB().fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    if toPrint:
        from sklearn.metrics import accuracy_score
        print(accuracy_score(y_test, y_predicted))
    else:
        print(list(y_predicted))

def LinearSVM(x_train, y_train, x_test, y_test, toPrint):
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(C=0.01, loss='hinge').fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    if toPrint:
        from sklearn.metrics import accuracy_score
        print(accuracy_score(y_test, y_predicted))
    else:
        print(list(y_predicted))

def SGDSVM(x_train, y_train, x_test, y_test, toPrint):
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier().fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    if toPrint:
        from sklearn.metrics import accuracy_score
        print(accuracy_score(y_test, y_predicted))
    else:
        print(list(y_predicted))

def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")

def LinearSVMFinal(x_train, y_train, x_test, y_test, toPrint=False):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    p = np.random.permutation(x_train.shape[0])

    x_train, y_train = x_train[p], y_train[p]
    train_size = int(0.8 * x_train.shape[0])

    x_train_train = x_train[:train_size, :]
    x_valid = x_train[train_size:, :]
    y_train_train = y_train[:train_size]
    y_valid = y_train[train_size:]

    C_poss = [1.0, 3.0, 5.0]
    accuracy = []
    for C in C_poss:
        if toPrint: print('C =', C)
        t = time.time()
        classifier = LinearSVC(C=C, loss='hinge').fit(x_train_train, y_train_train)
        y_predicted = classifier.predict(x_valid)
        accuracy.append(accuracy_score(y_valid, y_predicted))
        if toPrint:
            print(accuracy)
            print(time.time() - t)

    C = C_poss[accuracy.index(max(accuracy))]
    classifier = LinearSVC(C=C, loss='hinge').fit(x_train, y_train)
    y_predicted = np.array([int(y_) for y_ in classifier.predict(x_test)])
    if toPrint:
        print(accuracy_score(y_test, y_predicted))
    write_predictions(sys.argv[3], y_predicted)

def SGDFinal(x_train, y_train, x_test, y_test, toPrint=False):
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score

    p = np.random.permutation(x_train.shape[0])

    x_train, y_train = x_train[p], y_train[p]
    train_size = int(0.8 * x_train.shape[0])

    x_train_train = x_train[:train_size, :]
    x_valid = x_train[train_size:, :]
    y_train_train = y_train[:train_size]
    y_valid = y_train[train_size:]

    C_poss = [1e-6, 2e-6, 5e-6, 1e-5]
    accuracy = []
    for C in C_poss:
        if toPrint: print('alpha =', C)
        t = time.time()
        classifier = SGDClassifier(alpha=C, loss='hinge').fit(x_train_train, y_train_train)
        y_predicted = classifier.predict(x_valid)
        accuracy.append(accuracy_score(y_valid, y_predicted))
        if toPrint:
            print(accuracy)
            print(time.time() - t)

    C = C_poss[accuracy.index(max(accuracy))]
    classifier = SGDClassifier(alpha=C, loss='hinge').fit(x_train, y_train)
    y_predicted = np.array([int(y_) for y_ in classifier.predict(x_test)])
    if toPrint:
        print(accuracy_score(y_test, y_predicted))
    write_predictions(sys.argv[3], y_predicted)

# option =
# 1 if naive bayes
# 2 if svm with liblinear
# 3 if svm with sgd
# 4 if final liblinear
# 5 if final sgd
def main(option=1, toPrint=True):

    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]

    x_train, y_train = getXY(trainDataFile)
    x_test, y_test = getXY(testDataFile)
    x_train, y_train, x_test, y_test = transformData(x_train, y_train, x_test, y_test)

    if option == 1:
        NaiveBayes(x_train, y_train, x_test, y_test, toPrint)
    elif option == 2:
        LinearSVM(x_train, y_train, x_test, y_test, toPrint)
    elif option == 3:
        SGDSVM(x_train, y_train, x_test, y_test, toPrint)
    elif option == 4:
        LinearSVMFinal(x_train, y_train, x_test, y_test, toPrint)
    elif option == 5:
        SGDFinal(x_train, y_train, x_test, y_test, toPrint)
    else:
        raise Exception('Invalid option')

    return

if __name__ == '__main__':
    main(option=5, toPrint=False)
