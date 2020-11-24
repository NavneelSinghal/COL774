import sys
import csv
import numpy as np

# do something

def extractFeaturesAndLabels(trainDataFile, testDataFile):
    train = np.genfromtxt(trainDataFile, delimiter=',')
    test = np.genfromtxt(testDataFile, delimiter=',')
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    return x_train, y_train, x_test, y_test

def mainA():

    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]
    outputFile = sys.argv[3]

    x_train, y_train, x_test, y_test = extractFeaturesAndLabels(trainDataFile, testDataFile)

    x_train /= 255
    x_test /= 255

    # now filter the ones with y = 1
    train = [[], []]
    test = [[], []]
    # class 0 v/s 1
    for i in range(len(y_train)):
        if y_train[i] < 2:
            train[y_train[i]].append(x_train[i])
    # figure out the math and do stuff


def mainB():
    pass


if __name__ == '__main__':
    mainA()
