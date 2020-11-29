import sys
import os
import sklearn
import json
import numpy as np

def getXY(dataFile):
    data = []
    with open(dataFile) as data_:
        for line in data_:
            data.append(json.loads(line))
    x_data = [x['text'] for x in data]
    y_data = [int(x['stars']) for x in data]
    return x_data, y_data

def transformData(x_train, y_train, x_test, y_test):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    x_train = cv.fit_transform(x_train)
    y_train = np.array(y_train)
    x_test = cv.transform(x_test)
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

# TODO: implement validation set - takes a lot of time
def LinearSVM(x_train, y_train, x_test, y_test, toPrint):
    from sklearn.svm import LinearSVC
    classifier = LinearSVC().fit(x_train, y_train)
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

# option =
# 1 if naive bayes
# 2 if svm with liblinear
# 3 if svm with sgd
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
    else:
        raise Exception('Invalid option')

    return

if __name__ == '__main__':
    main(option=3, toPrint=False)
