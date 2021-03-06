import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('Agg')

import math
import numpy as np
import sys
from os.path import join, isfile

import warnings
warnings.filterwarnings("ignore")

# h_theta(x)
def h_theta(theta, x):
    return 1 / (1 + np.exp(-np.dot(theta, x)))

# computes gradient of the log likelihood
def log_likelihood_gradient(theta, x, y):
    llg = np.zeros(x[0].shape)
    for i in range(y.shape[0]):
        xi, yi = x[i], y[i][0]
        h_theta_xi = h_theta(theta, xi)
        llg += (yi - h_theta_xi) * xi
    return llg

# computes the hessian of the log likelihood given theta and x
def hessian(theta, x):
    h = np.zeros((x[0].shape[0], x[0].shape[0]))
    for xi in x:
        e = np.exp(-np.dot(theta, xi))
        h += e / (1 + e)**2 * np.outer(xi, xi)
    return -h

# performs logistic regression using newton's method
def logistic_regression(x, y):
    x = np.vstack((np.ones(x[0].shape[0]), x)).T
    y = y.T
    theta = np.full_like(x[0], 0)
    eps = 1e-7
    while True:
        llg = log_likelihood_gradient(theta, x, y)
        h = hessian(theta, x)
        diff = np.matmul(np.linalg.inv(h), llg)
        theta -= diff
        if np.linalg.norm(diff) < eps:
            break
    return theta

def main():

    # part A

    # read command-line arguments
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    part = sys.argv[3]

    # check for existence of input files
    for c in ['X', 'Y']:
        if not isfile(join(data_dir, 'logistic' + c + '.csv')):
            raise Exception('logistic' + c + '.csv not found')

    # read from csv file
    x = np.array(np.genfromtxt(join(data_dir, 'logisticX.csv'), delimiter=',')).T
    y = np.array([np.genfromtxt(join(data_dir, 'logisticY.csv'))])

    # normalisation
    x_mean = np.array([np.mean(x[0]), np.mean(x[1])])
    x[0] -= np.full_like(x[0], np.mean(x[0]))
    x[1] -= np.full_like(x[1], np.mean(x[1]))
    x_stddev = np.array([np.sqrt(np.sum(x[0] ** 2) / x[0].shape[0]), np.sqrt(np.sum(x[1] ** 2) / x[1].shape[0])])
    x[0] /= np.sqrt(np.sum(x[0] ** 2) / x[0].shape[0])
    x[1] /= np.sqrt(np.sum(x[1] ** 2) / x[1].shape[0])

    # perform the regression
    theta = logistic_regression(x, y)

    if part == 'a':
        output_file = open(join(out_dir, '3aoutput.txt'), mode='w')
        output_file.write('theta = ' + str(theta) + '\n')
        output_file.close()
        print('theta = ' + str(theta))
        return 0

    # part B

    fig3b, ax3b = plt.subplots()

    # filter on the basis of y-values
    x0, x1 = [], []
    for i in range(x[0].shape[0]):
        if y[0][i] == 0:
            x0.append([x[0][i], x[1][i]])
        else:
            x1.append([x[0][i], x[1][i]])
    x0 = np.array(x0).T
    x1 = np.array(x1).T
    ax3b.set_xlabel('x_0')
    ax3b.set_ylabel('x_1')

    # plotting both classes separately
    ax3b.scatter(x0[0] * x_stddev[0] + x_mean[0], x0[1] * x_stddev[1] + x_mean[1], c='red')
    ax3b.scatter(x1[0] * x_stddev[0] + x_mean[0], x1[1] * x_stddev[1] + x_mean[1], c='blue')

    # plotting the separator
    rx = np.arange(-2, 3)
    ry = (-theta[0] - theta[1] * rx) / theta[2]
    ax3b.plot(rx * x_stddev[0] + x_mean[0], ry * x_stddev[1] + x_mean[1])
    if part == 'b':
        fig3b.savefig(join(out_dir, 'regression_plot.png'))
        plt.show()
        return 0
    plt.close(fig3b)

if __name__ == '__main__':
    main()
