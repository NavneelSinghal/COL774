import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('Agg')

import math
import numpy as np
import sys
from os.path import join, isfile

def gda(x, y):
    x = x.T
    y = y.T
    phi, mu, sigma, M = 0, np.array([0., 0.]), 0, np.array([0, 0])
    m = y.shape[0]
    M[1] = np.sum(y)
    M[0] = m - M[1]
    phi = M[1] / m
    mu = np.array([np.sum(np.array([x[j] for j in range(m) if y[j] == i]), axis=0) / M[i] for i in range(2)])
    sigma = np.sum(np.array([np.outer(x[i] - mu[y[i]], x[i] - mu[y[i]]) for i in range(m)]), axis=0).astype(float) / m
    return phi, mu, sigma

def gda_general(x, y):
    x = x.T
    y = y.T
    phi, mu, sigma, M = 0, np.array([0., 0.]), 0, np.array([0, 0])
    m = y.shape[0]
    M[1] = np.sum(y)
    M[0] = m - M[1]
    phi = M[1] / m
    mu = np.array([np.sum(np.array([x[j] for j in range(m) if y[j] == i]), axis=0) / M[i] for i in range(2)])
    sigma = np.array([np.sum(np.array([np.outer(x[i] - mu[k], x[i] - mu[k]) for i in range(m) if y[i] == k]), axis=0) / M[k] for k in range(2)]).astype(float)
    return phi, mu, sigma

def main():


    # read command-line arguments
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    # check for existence of input files
    for c in ['x', 'y']:
        if not isfile(join(data_dir, 'q4' + c + '.dat')):
            raise Exception('q4' + c + '.dat not found')

    # read from csv file
    x = np.array(np.genfromtxt(join(data_dir, 'q4x.dat'))).T
    y = np.array([0 if yi == 'Alaska' else 1 for yi in np.loadtxt(join(data_dir, 'q4y.dat'), dtype=str)])

    # normalisation
    for i in range(2):
        x[i] -= np.full_like(x[i], np.mean(x[i]))
        x[i] /= np.sqrt(np.sum(x[i] ** 2) / x[i].shape[0])

    # part A

    phi, mu, sigma = gda(x, y)

    output_file = open(join(out_dir, '4aoutput.txt'), mode='w')
    output_file.write('phi = ' + str(phi) + '\n')
    output_file.write('mu[0] = ' + str(mu[0]) + '\n')
    output_file.write('mu[1] = ' + str(mu[1]) + '\n')
    output_file.write('sigma = \n' + str(sigma) + '\n')
    output_file.close()

    # part B, C

    fig4b, ax4b = plt.subplots()
    x0 = []
    x1 = []
    for i in range(y.shape[0]):
        if y[i] == 0:
            x0.append([x[0][i], x[1][i]])
        else:
            x1.append([x[0][i], x[1][i]])
    x0 = np.array(x0).T
    x1 = np.array(x1).T
    ax4b.scatter(x0[0], x0[1], c='red', s=6)
    ax4b.scatter(x1[0], x1[1], c='blue', s=6)

    # linear boundary
    sigma_inverse = np.linalg.inv(sigma)
    theta = np.array([0., 0., 0.])
    theta[0] = np.log(phi / (1 - phi))
    for i in range(2):
        mui = np.array([mu[i]])
        theta[0] += ((-1) ** i) * np.matmul(np.matmul(mui, sigma_inverse), mui.T)
    theta[1:] = np.matmul(np.array([mu[1] - mu[0]]), sigma_inverse)
    rx = np.arange(-3, 4)
    ry = (-theta[0] - theta[1] * rx) / theta[2]
    ax4b.plot(rx, ry)
    fig4b.savefig(join(out_dir, 'regression_plot.png'))

    # part D

    phi, mu, sigma = gda_general(x, y)

    output_file = open(join(out_dir, '4doutput.txt'), mode='w')
    output_file.write('phi = ' + str(phi) + '\n')
    output_file.write('mu[0] = ' + str(mu[0]) + '\n')
    output_file.write('mu[1] = ' + str(mu[1]) + '\n')
    output_file.write('sigma[0] = \n' + str(sigma[0]) + '\n')
    output_file.write('sigma[1] = \n' + str(sigma[1]) + '\n')
    output_file.close()

    # part E

    constant = np.log(phi / (1 - phi)) + np.log(np.linalg.det(sigma[0]) / np.linalg.det(sigma[1])) / 2
    linear = 0
    quadratic = 0
    for i in range(2):
        sigma_inverse = np.linalg.inv(sigma[i])
        mui = np.array([mu[i]])
        prod = np.matmul(mui, sigma_inverse)
        constant += ((-1) ** i) * np.matmul(prod, mui.T) / 2
        linear += ((-1) ** (i + 1)) * prod
        quadratic += ((-1) ** i) * sigma_inverse / 2
    constant = constant[0][0]
    linear = linear[0]
    # note that here x transposed is the feature vector (as x is a row vector)
    # and similarly mu[i] is also a row vector, which explains the equations above
    # equation is x * quadratic * x.T + linear * x.T + constant = 0
    Z = 0
    X, Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    Z += quadratic[0, 0] * (X ** 2) + (quadratic[0, 1] + quadratic[1, 0]) * X * Y + (quadratic[1, 1]) * (Y ** 2)
    Z += linear[0] * X + linear[1] * Y
    Z += constant
    ax4b.contour(X, Y, Z, 0)
    fig4b.savefig(join(out_dir, 'regression_plot_2.png'))

    # part F - in the report


if __name__ == '__main__':
    main()
