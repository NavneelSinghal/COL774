import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('Agg')

import math
import numpy as np
import sys
import os
from os.path import join, isfile

import warnings
warnings.filterwarnings("ignore")

def stochastic_gradient_descent(x, y, b, eta=1e-3, epsilon=1e-2, k=10): # b = batch size

    # number of examples
    m = x.shape[0]

    # theta: parameters
    # prev_loss: previous loss value
    # losses: list of all losses
    # thetas: list of all thetas
    # current_streak: max number of consecutive updates satisfying convergence condition
    # ran_iterations: number of iterations yet
    theta = np.array([0., 0., 0.])
    prev_loss = 1000000
    losses = [prev_loss]
    thetas = [theta]
    current_streak = 0
    ran_iterations = 0

    # this is in case we want to ever change to Andrew Ng's convergence criteria
    max_iterations = min(1000, 5 * m // b)
    cumulative_loss = 0
    previous_loss = 0

    # for preventing timeout - for smaller epsilons
    import time
    start_time = time.time()

    while True:

        current_time = time.time()
        # for preventing even accidental infinite looping
        # note that it converges before that with a high probability with parameter 2e-3
        if current_time > start_time + 600:
            break

        # shuffling randomly
        p = np.random.permutation(x.shape[0])
        x, y = x[p], y[p]

        done = False

        for i in range(0, b * (m // b), b):

            # in what follows,
            # grad = gradient value for this mini-batch
            # loss = loss value for this mini-batch

            # vectorized version below
            #grad = np.array([0., 0., 0.])
            #loss = 0
            #for j in range(b):
            #    diff_error = np.dot(theta, x[i + j]) - y[i + j]
            #    grad += diff_error * x[i + j]
            #    loss += diff_error ** 2
            #loss /= (2 * b)
            #grad /= b

            x_batch = x[i : i + b].T
            y_batch = y[i : i + b]
            diff_errors = np.tile(np.matmul(theta[np.newaxis, :], x_batch)[0] - y_batch, (3, 1))
            grad = np.sum(diff_errors * x_batch, axis=1) / b
            loss = np.sum(diff_errors[0] ** 2) / (2 * b)

            # updating the value of theta, and storing the theta and the loss
            theta = theta - eta * grad
            ran_iterations += 1
            thetas.append(theta)
            losses.append(loss)

            # for convergence criteria involving average loss as in Andrew Ng's video
            cumulative_loss += loss
            if ran_iterations % max_iterations == 0:
                cumulative_loss /= max_iterations
                losses.append(cumulative_loss)
                if abs(cumulative_loss - previous_loss) < epsilon:
                    done = True
                    break
                previous_loss = cumulative_loss
                cumulative_loss = 0

            #if np.linalg.norm(thetas[-1] - thetas[-2]) < epsilon: # alternative convergence criterion
            #if abs(loss - prev_loss) < epsilon:
            #    current_streak += 1
            #else:
            #    current_streak = 0

            prev_loss = loss

            #done = done or current_streak >= k

            if done:
                break

        if done:
            break

    return theta, thetas[1:], ran_iterations, losses[1:]

def main():

    # read command-line arguments
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    part = sys.argv[3]

    # part A

    mu = np.array([3., -1.])
    sigma = np.array([np.sqrt(4), np.sqrt(4)])
    m = 1000000
    x = [None] * 3
    x[0] = np.ones(m)
    for i in range(2):
        x[i + 1] = np.random.normal(mu[i], sigma[i], m)
    theta_gen = np.array([3., 1., 2.])
    x = np.array(x).T
    y = np.array([np.dot(xi, theta_gen) for xi in x]) + np.random.normal(0, np.sqrt(2), m)
    if part == 'a':
        print('x:\n' + str(x))
        print('y:\n' + str(y))
        out_file = open(join(out_dir, '2a.txt'), 'w')
        out_file.write('x:\n' + str(x) + '\n')
        out_file.write('y:\n' + str(y) + '\n')
        out_file.close()
        return 0

    # part B

    theta = [None] * 4
    thetas = [None] * 4
    iterations = [None] * 4
    loss = [None] * 4

    import pickle

    pickle_file = None
    pickle_found = False
    # if we already have run this program before, just use the values stored
    for dirpath, dirs, files in os.walk(join(out_dir, '..')):
        for filename in files:
            fname = os.path.join(dirpath,filename)
            if fname.endswith('store.pickle'):
                pickle_file = fname
                pickle_found = True
                break
    #if isfile(join(out_dir, 'store.pickle')):
    if pickle_found:
        theta, thetas, iterations, loss = pickle.load(open(pickle_file, 'rb'))
    else:
        epsilons = [5e-5] * 4 # just in case we decide to try out till time runs out
        #epsilons = [2e-3] * 4 # default value of epsilons
        for i, b in enumerate([1, 100, 10000, 1000000]):
            # uncomment if you need verbose training statistics
            import time
            old_time = time.time()
            if part == 'b':
                print('starting training model for batch size ' + str(b))
            theta[i], thetas[i], iterations[i], loss[i] = stochastic_gradient_descent(x, y, b, epsilon=epsilons[i], k=min(3, m // b))
            new_time = time.time()
            if part == 'b':
                print('batch size ' + str(b) + ' done')
                print(theta[i])
                print('time taken: ' + str(new_time - old_time))
            #print(theta[i])
            #print(iterations[i])
            #print(loss[i][-1])
        pickle.dump((theta, thetas, iterations, loss), open(join(out_dir, 'store.pickle'), 'wb'))


    if part == 'b':
        out_file = open(join(out_dir, '2b.txt'), 'w')
        out_file.write('values of theta for each batch size are (as row vectors):\n' + str(np.array(theta)) + '\n')
        out_file.write('iterations taken for each:\n' + str(np.array(iterations)) + '\n')
        out_file.close()
        print('values of theta for each batch size are (as row vectors):\n' + str(np.array(theta)))
        print('iterations taken for each:\n' + str(np.array(iterations)))
        return 0

    # part C - subjective part in the report

    # reading data from the file into a convenient format
    x = []
    data_file = open(join(data_dir, 'q2test.csv'))
    lines = data_file.readlines()[1:]
    for line in lines:
        x.append(np.fromstring(line, dtype=float, sep=','))
    x = np.hstack((np.ones((len(x), 1)), np.array(x))).T
    y = np.array([x[3]]).T
    x = x[:3].T

    errors = []
    theta.append(theta_gen)
    # the last element of errors corresponds to the error due to prediction using the original hypothesis

    new_m = y.shape[0]
    for t in theta:
        error = 0
        for i in range(new_m):
            error += (np.dot(t, x[i]) - y[i][0]) ** 2
        error /= 2 * new_m
        errors.append(error)


    if part == 'c':
        out_file = open(join(out_dir, '2c.txt'), 'w')
        out_file.write('errors (last is original hypothesis, rest ordered wrt batch size): ' + str(errors))
        out_file.close()
        print('errors (last is original hypothesis, rest ordered wrt batch size): ' + str(errors))
        return 0

    # part D - simple plotting

    for i, t in enumerate(thetas):
        t = np.array(t).T
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim3d([-0.5, 3.5])
        ax.set_xlabel('Theta_0')
        ax.set_ylim3d([-0.5, 1.5])
        ax.set_ylabel('Theta_1')
        ax.set_zlim3d([-0.5, 2.5])
        ax.set_zlabel('Theta_2')
        plot = ax.plot([t[0, 0]], [t[1, 0]], [t[2, 0]])
        # actually the problem didn't ask for the animation
        # update function for the animation
        def update(nums):
            plot[0].set_data(t[0:2, :nums])
            plot[0].set_3d_properties(t[2, :nums])
            return plot
        # performing the animation
        #anim = animation.FuncAnimation(fig, update, t.shape[1], interval=200, blit=True)
        update(t.shape[1])
        if part == 'd':
            fig.savefig(join(out_dir, '2d' + str(i + 1) + 'last_frame.png'))
            plt.show()
        plt.close(fig)

    return 0

if __name__ == '__main__':
    main()
