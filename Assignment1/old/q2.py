import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#matplotlib.use('Agg')

import math
import numpy as np
import sys
from os.path import join, isfile

def stochastic_gradient_descent(x, y, b, eta=1e-3, epsilon=1e-2, k=10): # b = batch size

    m = x.shape[0]

    theta = np.array([0., 0., 0.])
    losses = [0]
    thetas = [theta]
    #losses = []
    current_streak = 0
    ran_iterations = 0
    #max_iterations = min(1000, m // b)
    #cumulative_loss = 0
    #previous_loss = 0
    import time
    start_time = time.time()

    while True:

        current_time = time.time()
        if current_time > start_time + 90:
            break # for preventing even accidental infinite looping, even though I'm sure it converges before that with a high probability with parameter 2e-3

        # shuffling randomly
        p = np.random.permutation(x.shape[0])
        x, y = x[p], y[p]

        done = False

        for i in range(0, b * (m // b), b):

            grad = np.array([0., 0., 0.])
            loss = 0

            x_batch = x[i : i + b].T
            y_batch = y[i : i + b]
            diff_errors = np.tile(np.matmul(theta[np.newaxis, :], x_batch)[0] - y_batch, (3, 1))
            grad = np.sum(diff_errors * x_batch, axis=1)
            loss = np.sum(diff_errors[0] ** 2)

            # vectorized version above
            #for j in range(b):
            #    diff_error = np.dot(theta, x[i + j]) - y[i + j]
            #    grad += diff_error * x[i + j]
            #    loss += diff_error ** 2

            loss /= (2 * b)
            grad /= b
            theta = theta - eta * grad
            ran_iterations += 1
            thetas.append(theta)
            losses.append(loss)

            # for convergence criteria involving average loss
            '''
            cumulative_loss += loss
            if ran_iterations % max_iterations == 0:
                cumulative_loss /= max_iterations
                losses.append(cumulative_loss)
                if abs(cumulative_loss - previous_loss) < epsilon:
                    done = True
                    break
                previous_loss = cumulative_loss
                cumulative_loss = 0
            '''

            #if np.linalg.norm(thetas[-1] - thetas[-2]) < epsilon: # alternative convergence criterion
            if abs(losses[-1] - losses[-2]) < epsilon:
                current_streak += 1
            else:
                current_streak = 0

            done = done or current_streak >= k

            if done:
                break

        if done:
            break

    return theta, thetas[1:], ran_iterations, losses[1:]

def main():

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

    # part B

    theta = [None] * 4
    thetas = [None] * 4
    iterations = [None] * 4
    loss = [None] * 4

    # read command-line arguments
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    import pickle

    if isfile(join(out_dir, 'store.pickle')):
        theta, thetas, iterations, loss = pickle.load(open(join(out_dir, 'store.pickle'), 'rb'))
    else:
        #epsilons = [1e-4] * 4 #just in case we decide to try out till time runs out
        epsilons = [2e-3] * 4
        for i, b in enumerate([1, 100, 10000, 1000000]):
            #import time
            #old_time = time.time()
            theta[i], thetas[i], iterations[i], loss[i] = stochastic_gradient_descent(x, y, b, epsilon=epsilons[i], k=min(3, m // b))
            #print(str(b) + ' done')
            #print(theta[i])
            #new_time = time.time()
            #print('time taken: ' + str(new_time - old_time))
            #print('done')
            #print(theta[i])
            #print(iterations[i])
            #print(loss[i][-1])
        pickle.dump((theta, thetas, iterations, loss), open(join(out_dir, 'store.pickle'), 'wb'))

    out_file = open(join(out_dir, '2b.txt'), 'w')
    out_file.write('theta for each iteration are ' + str(np.array(theta)))
    out_file.write('\niterations taken for each: ' + str(np.array(iterations)))
    out_file.close()

    # part C - subjective part in the report

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

    new_m = y.shape[0]
    for t in theta:
        error = 0
        for i in range(new_m):
            error += (np.dot(t, x[i]) - y[i][0]) ** 2
        error /= 2 * new_m
        errors.append(error)

    out_file = open(join(out_dir, '2c.txt'), 'w')
    out_file.write('errors (last is original hypothesis, rest ordered wrt batch size): ' + str(errors))
    out_file.close()


    # last element of errors corresponds to the error due to prediction using the original hypothesis

    # part D

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
        plot = ax.plot(t[0, 0], t[1, 0], t[2, 0])
        def update(nums):
            plot[0].set_data(t[0:2, :nums])
            plot[0].set_3d_properties(t[2, :nums])
            return plot
        # performing the animation
        anim = animation.FuncAnimation(fig, update, t.shape[1], blit=True)
        #fig.show()
        update(t.shape[1])
        fig.savefig(join(out_dir, '2d' + str(i + 1) + 'last_frame.png'))



if __name__ == '__main__':
    main()
