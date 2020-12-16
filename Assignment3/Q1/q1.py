from collections import defaultdict, deque
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
import sys
import math

# replace by math.inf later
NODE_MAX = 200000
NODE_MAX = math.inf

def freq_dict(l):
    d = defaultdict(int)
    for i in l:
        d[i] += 1
    return d

def most_frequent(d):
    return max(d.items(), key=itemgetter(1))[0]

def choose_attribute(x, is_bool):
    j_best, split_value, min_entropy = -1, -1, math.inf
    y = x[:, -1]
    for j in range(len(is_bool)):
        w, med = x[:, j], 0.5
        if not is_bool[j]: med = np.median(w)
        mn, mx = np.min(w), np.max(w)
        if mn == mx or med == mx: continue
        y_split = [y[w <= med], y[w > med]]
        entropy, p = 0, 1 / len(y)
        for y_ in y_split:
            h, prob = 0, 1 / len(y_)
            _, counts = np.unique(y_, return_counts=True)
            for value in counts:
                temp = prob * value
                h += temp * math.log(1 / temp)
            entropy += p * len(y_) * h
        if entropy < min_entropy:
            min_entropy = entropy
            j_best = j
            split_value = med
    if j_best == -1:
        return -1, None, None, None, None, None
    left = x[x[:, j_best] <= split_value]
    right = x[x[:, j_best] > split_value]
    return j_best, split_value, left, right

class Node:
    """
    Node class for the decision tree

    Data:
    self.left, self.right: left and right children
    self.is_leaf: boolean
    self.attribute_num: attribute number being split on - if self.is_leaf is False
    self.class_freq: class frequencies if self.is_leaf is True
    self.cl: class decision if self.is_leaf is True
    self.x: data associated to this node if self.is_leaf is True (used while growing tree)
    """

    # by default each node is a leaf
    def __init__(self, x):
        self.left = None
        self.right = None
        self.attribute_num = -1
        self.is_leaf = True
        self.x = x
        self.class_freq = freq_dict(x[:, -1])
        self.cl = most_frequent(self.class_freq)
        self.split_value = None

class DecisionTree:

    # D is a numpy array, last col is y
    # is_bool: list of booleans: True if data is boolean, False if data is int
    # threshold: threshold for training accuracy
    def __init__(self,
                 D_train=None,
                 D_test=None,
                 D_valid=None,
                 is_bool=None,
                 threshold=1.0,
                 prediction_frequency=1000,
                 pruning=False):
        self.train_accuracies = []
        self.test_accuracies = []
        self.valid_accuracies = []
        if D_train is not None:
            self.grow_tree(
                    D_train=D_train,
                    D_test=D_test,
                    D_valid=D_valid,
                    is_bool=is_bool,
                    threshold=threshold,
                    prediction_frequency=prediction_frequency,
                    pruning=pruning)
        else:
            self.root = None

    def predict(self, D_test):
        predicted = []
        for x in D_test:
            node = self.root
            while not node.is_leaf:
                if x[node.attribute_num] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            predicted.append(node.cl)
        return np.array(predicted)

    def grow_tree(self,
                  D_train,
                  D_test,
                  D_valid,
                  is_bool,
                  threshold,
                  prediction_frequency,
                  pruning):
        # empty data
        if len(D_train) == 0:
            raise Exception('Empty data')

        self.root = Node(x=D_train)

        q = deque()
        q.appendleft(self.root)

        node_list = []
        node_list.append(self.root)

        predictions_completed = 0
        train_accuracy = 0
        test_accuracy = 0
        valid_accuracy = 0
        total_nodes = 1
        y_train = D_train[:, -1]
        y_test = D_test[:, -1]
        y_valid = D_valid[:, -1]

        while train_accuracy < threshold and q and total_nodes < NODE_MAX:

            node = q.pop()

            # if node is pure
            if len(node.class_freq) == 1:

                node.x = None

            else:

                j, node.split_value, left_x, right_x = choose_attribute(node.x, is_bool)

                if j == -1:
                    node.x = None
                    continue

                node.x, node.cl, node.class_freq = None, None, None
                node.attribute_num = j
                node.is_leaf = False

                node.left = Node(left_x)
                node.right = Node(right_x)

                node_list.append(node.left)
                node_list.append(node.right)

                q.appendleft(node.left)
                q.appendleft(node.right)

                total_nodes += 2

                if total_nodes > (predictions_completed * prediction_frequency):
                    #print('total nodes expanded', total_nodes)
                    predictions_completed += 1
                    train_pred = self.predict(D_train[:, :-1])
                    test_pred = self.predict(D_test[:, :-1])
                    valid_pred = self.predict(D_valid[:, :-1])
                    train_accuracy = len(y_train[y_train == train_pred]) / len(train_pred)
                    test_accuracy = len(y_test[y_test == test_pred]) / len(test_pred)
                    valid_accuracy = len(y_valid[y_valid == valid_pred]) / len(valid_pred)
                    self.train_accuracies.append(train_accuracy)
                    self.test_accuracies.append(test_accuracy)
                    self.valid_accuracies.append(valid_accuracy)

        # finally discard all data in leaf nodes
        for node in node_list:
            node.x = None

        if not pruning:
            return

def main():

    train = np.loadtxt(sys.argv[1], delimiter=',', skiprows=2)
    test = np.loadtxt(sys.argv[2], delimiter=',', skiprows=2)
    valid = np.loadtxt(sys.argv[3], delimiter=',', skiprows=2)

    is_bool = [(False if i < 10 else True) for i in range(54)]
    prediction_frequency = 2000

    decision_tree = DecisionTree(
            D_train=train,
            D_test=test,
            D_valid=valid,
            is_bool=is_bool,
            threshold=1.0,
            prediction_frequency=prediction_frequency)

    x = [i * prediction_frequency for i in range(len(decision_tree.train_accuracies))]
    plt.plot(x, decision_tree.train_accuracies)
    plt.plot(x, decision_tree.test_accuracies)
    plt.plot(x, decision_tree.valid_accuracies)
    plt.show()

    pass

if __name__ == '__main__':
    main()
