import sys
import csv
import numpy as np
import cvxopt as co
import pickle
import time
import os

eps = 1e-5
inf = 1e9

def extractFeaturesAndLabels(dataFile):
    data = np.genfromtxt(dataFile, delimiter=',')
    x, y = data[:, :-1], data[:, -1]
    return x, y

def filterOut(x, y, class1=0, class2=1):
    tx = [[], []]
    for i in range(len(y)):
        if int(y[i]) == class1:
            tx[0].append(x[i])
        elif int(y[i]) == class2:
            tx[1].append(x[i])
    y = np.array([-1.0] * len(tx[0]) + [1.0] * len(tx[1]))
    x = np.array(tx[0] + tx[1])
    return x, y

def computeParametersLinearKernel(x, y, c=1.0):
    z = [None] * len(y)
    for i in range(len(y)):
        z[i] = x[i] * y[i]
    z = np.array(z)
    m = len(y)
    P = co.matrix(np.matmul(z, z.T))
    q = co.matrix([-1.0] * m)
    G = co.matrix(np.vstack((-1 * np.identity(m), np.identity(m))))
    A = co.matrix([[y[i]] for i in range(m)])
    b = co.matrix([0.0])
    h = co.matrix([0.0] * m + [c] * m)
    sol = co.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    assert sol['status'] == 'optimal'
    alpha = sol['x']
    w = 0
    for i in range(m):
        w += alpha[i] * z[i]
    supportVectorIndices = [i for i in range(m) if abs(alpha[i]) > eps]
    mna, mxa = inf, -inf
    bSum = 0.0
    bNum = 0
    for i in range(m):
        if abs(c - alpha[i]) < eps:
            continue
        val = y[i] - np.dot(w, x[i])
        if abs(alpha[i]) > eps:
            bNum += 1
            bSum += val
        if int(y[i]) == -1:
            mna = min(mna, val)
        else:
            mxa = max(mxa, val)
    b = (mna + mxa) / 2
    if bNum != 0:
        b = bSum / bNum
    return (w, b, supportVectorIndices)

# a[i, j] = norm of difference of i, jth elements
def squaredDistanceMatrix(x, y, same=False):
    if same:
        squares = np.einsum('ij,ij->i', x, x)
        squares_fill = np.tile(squares, (squares.shape[0], 1))
        return squares_fill + squares_fill.T - 2 * np.matmul(x, x.T)
    squares_x = np.einsum('ij,ij->i', x, x)
    squares_y = np.einsum('ij,ij->i', y, y)
    return np.tile(squares_y, (squares_x.shape[0], 1)) + np.tile(squares_x, (squares_y.shape[0], 1)).T - 2 * np.matmul(x, y.T)

def squaredDistanceMatrixOld(x, y):
    squares_x = np.einsum('ij,ij->i', x, x)
    squares_y = np.einsum('ij,ij->i', y, y)
    squares_x_fill = np.vstack(tuple([squares_x] * squares_y.shape[0]))
    return np.vstack(tuple([squares_y] * squares_x.shape[0])) + squares_x_fill.T - 2 * np.matmul(x, y.T)#2 * np.einsum('ik,jk->ij', x, y)

def computeParametersGaussianKernel(x, y, c=1.0, gamma=0.05):
    m = len(y)
    squares = np.exp(-gamma * squaredDistanceMatrix(x, x, same=True))
    #squares = np.exp(-gamma * squaredDistanceMatrixOld(x, x))
    P = co.matrix(squares * np.outer(y, y))
    q = co.matrix([-1.0] * m)
    G = co.matrix(np.vstack((-1 * np.identity(m), np.identity(m))))
    A = co.matrix([[y[i]] for i in range(m)])
    b = co.matrix([0.0])
    h = co.matrix([0.0] * m + [c] * m)
    sol = co.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    assert sol['status'] == 'optimal'
    alpha = (np.array(sol['x']).T)[0]
    supportVectorIndices = [i for i in range(m) if abs(alpha[i]) > eps]
    values = np.einsum('i,i,ij->j', alpha, y, squares)
    assert len(values) == len(y)
    mna, mxa = inf, -inf
    bSum = 0.0
    bNum = 0
    for i in range(m):
        if abs(c - alpha[i]) < eps:
            continue
        val = y[i] - values[i]
        if abs(alpha[i] > eps):
            bSum += val
            bNum += 1
        if int(y[i]) == -1:
            mna = min(mna, val)
        else:
            mxa = max(mxa, val)
    b = (mna + mxa) / 2
    if bNum != 0:
        b = bSum / bNum
    return (b, alpha, supportVectorIndices)

def accuracyLinear(x, y, w, b):
    m = len(y)
    misses = 0
    for i in range(m):
        y_predicted = 1 if (np.dot(w, x[i]) + b) >= 0.0 else -1
        if y_predicted != int(y[i]):
            misses += 1
    return (1 - misses / m)

# note that in this function, instead of x_train, y_train, we can use x_sv, y_sv
def testGaussian(x, alpha, b, x_train, y_train, gamma=0.05):
    #squares = squaredDistanceMatrix(x_train, x)
    #squares *= -gamma
    #squares = np.exp(squares)
    values = np.einsum('i,ij->j', alpha * y_train, np.exp(-gamma * squaredDistanceMatrix(x_train, x))) + b
    return [1 if v >= 0.0 else -1 for v in values]

def accuracyGaussian(x, y, alpha, b, x_train, y_train, gamma=0.05):
    y_predicted = testGaussian(x, alpha, b, x_train, y_train, gamma)
    m = len(y)
    misses = 0
    for i in range(m):
        if y_predicted[i] != int(y[i]):
            misses += 1
    return (1 - misses / m)

def mainA(linearKernel=True, cl=0):

    validDataFile = None
    x_valid = None
    y_valid = None
    foundValid = False

    if len(sys.argv) > 3:
        foundValid = True

    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]
    if foundValid:
        validDataFile = sys.argv[3]

    x_train, y_train = extractFeaturesAndLabels(trainDataFile)
    x_test, y_test = extractFeaturesAndLabels(testDataFile)
    if foundValid:
        x_valid, y_valid = extractFeaturesAndLabels(validDataFile)

    x_train /= 255
    x_test /= 255
    if foundValid:
        x_valid /= 255

    x_train, y_train = filterOut(x_train, y_train, cl, (cl + 1) % 10)
    x_test, y_test = filterOut(x_test, y_test, cl, (cl + 1) % 10)
    if foundValid:
        x_valid, y_valid = filterOut(x_valid, y_valid, cl, (cl + 1) % 10)

    #print('training vectors:', len(x_train))
    #print('testing vectors:', len(x_test))

    if linearKernel:
        print('Linear Kernel results:')
        w, b, supportVectorIndices = computeParametersLinearKernel(x_train, y_train)

        # cross verification
        #from sklearn import svm
        #clf = svm.SVC(kernel='linear', C=1.0)
        #clf.fit(x_train, y_train)
        #print('b for sklearn:', clf.intercept_[0])
        #print('number of support vectors for sklearn:', len(clf.support_vectors_))

        #supportVectors = x_train[supportVectorIndices].tolist()
        #print('support vectors:', len(supportVectors))
        #for supportVector in supportVectors:
        #    print([int(x * 1e3) / 1e3 for x in supportVector])
        #print('w:', [int(x * 1e3) / 1e3 for x in w])
        #print('b:', b)
        print('accuracy on train:', accuracyLinear(x_train, y_train, w, b))
        print('accuracy on test:', accuracyLinear(x_test, y_test, w, b))
        if foundValid:
            print('accuracy on validation:', accuracyLinear(x_valid, y_valid, w, b))

    else:
        print('Gaussian Kernel results:')
        b, alpha, supportVectorIndices = computeParametersGaussianKernel(x_train, y_train)

        # cross verification
        #from sklearn import svm
        #clf = svm.SVC(C=1.0, gamma=0.05)
        #clf.fit(x_train, y_train)
        #print('b for sklearn:', clf.intercept_[0])
        #print('number of support vectors for sklearn:', len(clf.support_vectors_))

        #supportVectors = x_train[supportVectorIndices].tolist()
        #print('support vectors:', len(supportVectors))
        #for supportVector in supportVectors:
        #    print([int(x * 1e3) / 1e3 for x in supportVector])
        #print('b:', b)
        #print('alpha:', alpha)
        x_sv = x_train[supportVectorIndices]
        y_sv = y_train[supportVectorIndices]
        alpha_sv = alpha[supportVectorIndices]
        print('accuracy on train:', accuracyGaussian(x_train, y_train, alpha_sv, b, x_sv, y_sv))
        print('accuracy on test:', accuracyGaussian(x_test, y_test, alpha_sv, b, x_sv, y_sv))
        if foundValid:
            print('accuracy on validation:', accuracyGaussian(x_valid, y_valid, alpha_sv, b, x_sv, y_sv))

def splitData(x, y):
    output = [[] for i in range(10)]
    for i in range(len(y)):
        output[int(y[i])].append(x[i])
    return [np.array(w) for w in output]

def efficientComputeParametersGaussianKernel(x, y, c=1.0, gamma=0.05):
    m = len(y)
    squares = np.exp(-gamma * squaredDistanceMatrix(x, x, True))
    P = co.matrix(squares * np.outer(y, y))
    q = co.matrix([-1.0] * m)
    G = co.matrix(np.vstack((-1 * np.identity(m), np.identity(m))))
    A = co.matrix([[y[i]] for i in range(m)])
    b = co.matrix([0.0])
    h = co.matrix([0.0] * m + [c] * m)
    sol = co.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    assert sol['status'] == 'optimal'
    alpha = (np.array(sol['x']).T)[0]
    values = np.einsum('i,i,ij->j', alpha, y, squares)
    assert len(values) == len(y)
    mna, mxa = inf, -inf
    bSum = 0.0
    bNum = 0
    for i in range(m):
        if abs(c - alpha[i]) < eps:
            continue
        val = y[i] - values[i]
        if abs(alpha[i] > eps):
            bSum += val
            bNum += 1
        if int(y[i]) == -1:
            mna = min(mna, val)
        else:
            mxa = max(mxa, val)
    b = (mna + mxa) / 2
    if bNum != 0:
        b = bSum / bNum
    supportVectorIndices = [i for i in range(m) if abs(alpha[i]) > eps]
    return (b, alpha[supportVectorIndices], supportVectorIndices)

def mainB():

    validDataFile = None
    x_valid = None
    x_valid_old = None
    y_valid = None
    foundValid = False

    if len(sys.argv) > 3:
        foundValid = True

    trainDataFile = sys.argv[1]
    testDataFile = sys.argv[2]
    if foundValid:
        validDataFile = sys.argv[3]

    x_train, y_train = extractFeaturesAndLabels(trainDataFile)
    x_test, y_test = extractFeaturesAndLabels(testDataFile)
    if foundValid:
        x_valid, y_valid = extractFeaturesAndLabels(validDataFile)

    x_train /= 255
    x_train_old = x_train
    x_test /= 255
    x_test_old = x_test
    if foundValid:
        x_valid /= 255
        x_valid_old = x_valid

    # use splitData instead
    x_train = splitData(x_train, y_train)
    x_test = splitData(x_test, y_test)
    if foundValid:
        x_valid = splitData(x_valid, y_valid)

    b = [[None for i in range(10)] for j in range(10)]
    alpha_sv = [[None for i in range(10)] for j in range(10)]
    sv_indices = [[None for i in range(10)] for j in range(10)]
    if os.path.isfile(os.path.join('.', 'multiclass.pickle')):
        b, alpha_sv, sv_indices = pickle.load(open('multiclass.pickle', 'rb'))
    else:
        t = time.time()
        for i in range(10):
            for j in range(i + 1, 10):
                #print(time.time() - t)
                #print('starting', i, 'and', j)
                x_concat = np.concatenate((x_train[i], x_train[j]))
                y_concat = np.concatenate((-1 * np.ones(len(x_train[i])), np.ones(len(x_train[j]))))
                b[i][j], alpha_sv[i][j], sv_indices[i][j] = efficientComputeParametersGaussianKernel(x_concat, y_concat)
        print('training time:', time.time() - t)
        pickle.dump((b, alpha_sv, sv_indices), open('multiclass.pickle', 'wb'))

    # test prediction
    t = time.time()
    y_test_prediction_freq = [[0 for i in range(10)] for j in range(len(y_test))]
    for i in range(10):
        for j in range(i + 1, 10):
            supportVectorIndices = sv_indices[i][j]
            x_sv = np.concatenate((x_train[i], x_train[j]))[supportVectorIndices]
            y_sv = np.concatenate((-1 * np.ones(len(x_train[i])), np.ones(len(x_train[j]))))[supportVectorIndices]
            #print('starting test prediction on classifier', i, j)
            y_predicted = testGaussian(x_test_old, alpha_sv[i][j], b[i][j], x_sv, y_sv, gamma=0.05)
            for k in range(len(x_test_old)):
                pred = y_predicted[k]
                if pred == -1:
                    y_test_prediction_freq[k][i] += 1
                else:
                    y_test_prediction_freq[k][j] += 1
    y_predicted = []
    for k in range(len(y_test)):
        y_predicted.append(y_test_prediction_freq[k].index(max(y_test_prediction_freq[k])))
    print('test prediction time:', time.time() - t)
    confusion_matrix = np.array([[0 for i in range(10)] for j in range(10)])
    for k in range(len(y_test)):
        confusion_matrix[y_predicted[k]][int(y_test[k])] += 1
    print('test confusion matrix:')
    print(confusion_matrix)
    correct = np.trace(confusion_matrix)
    total = np.sum(confusion_matrix)
    print('test accuracy:', correct / total)


    if foundValid:
        # validation prediction
        t = time.time()
        y_valid_prediction_freq = [[0 for i in range(10)] for j in range(len(y_valid))]
        for i in range(10):
            for j in range(i + 1, 10):
                supportVectorIndices = sv_indices[i][j]
                x_sv = np.concatenate((x_train[i], x_train[j]))[supportVectorIndices]
                y_sv = np.concatenate((-1 * np.ones(len(x_train[i])), np.ones(len(x_train[j]))))[supportVectorIndices]
                y_predicted = testGaussian(x_valid_old, alpha_sv[i][j], b[i][j], x_sv, y_sv, gamma=0.05)
                for k in range(len(x_valid_old)):
                    pred = y_predicted[k]
                    if pred == -1:
                        y_valid_prediction_freq[k][i] += 1
                    else:
                        y_valid_prediction_freq[k][j] += 1
        y_predicted = []
        for k in range(len(y_valid)):
            y_predicted.append(y_valid_prediction_freq[k].index(max(y_valid_prediction_freq[k])))
        print('validation prediction time:', time.time() - t)
        confusion_matrix = np.array([[0 for i in range(10)] for j in range(10)])
        for k in range(len(y_valid)):
            confusion_matrix[y_predicted[k]][int(y_valid[k])] += 1
        print('validation confusion matrix:')
        print(confusion_matrix)
        correct = np.trace(confusion_matrix)
        total = np.sum(confusion_matrix)
        print('validation accuracy:', correct / total)

    # cross verification
    from sklearn import svm, metrics
    clf = svm.SVC(C=1.0, gamma=0.05)

    if os.path.isfile(os.path.join('.', 'multiclass_sklearn.pickle')):
        clf = pickle.load(open('multiclass_sklearn.pickle', 'rb'))
    else:
        t = time.time()
        clf.fit(x_train_old, y_train)
        print('sklearn training time:', time.time() - t)
        pickle.dump(clf, open('multiclass_sklearn.pickle', 'wb'))

    t = time.time()
    test_predictions_sklearn = clf.predict(x_test_old)
    print('sklearn test prediction time:', time.time() - t)
    print('sklearn test confusion matrix:')
    confusion_matrix = np.array([[0 for i in range(10)] for j in range(10)])
    for k in range(len(y_test)):
        confusion_matrix[int(test_predictions_sklearn[k])][int(y_test[k])] += 1
    print(confusion_matrix)
    print('sklearn test accuracy:', np.trace(confusion_matrix) / np.sum(confusion_matrix))

    if foundValid:
        t = time.time()
        valid_predictions_sklearn = clf.predict(x_valid_old)
        print('sklearn validation prediction time:', time.time() - t)
        print('sklearn validation confusion matrix:')
        confusion_matrix = np.array([[0 for i in range(10)] for j in range(10)])
        for k in range(len(y_valid)):
            confusion_matrix[int(valid_predictions_sklearn[k])][int(y_valid[k])] += 1
        print(confusion_matrix)
        print('sklearn validation accuracy:', np.trace(confusion_matrix) / np.sum(confusion_matrix))

def mainBCrossValidation():

    pass

if __name__ == '__main__':
    #mainA(True)
    #mainA(False)
    mainB()
