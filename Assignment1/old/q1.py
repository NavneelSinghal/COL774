import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#matplotlib.use('Agg')

import numpy as np
import sys
from os.path import join, isfile

def grad_desc(x, y, eta=1e-2, epsilon=1e-15):

    # eta = learning rate
    # epsilon = error
    # m = number of examples
    m = x.shape[0]

    # x[i][0] = 1 for intercept term
    x = np.hstack((np.ones(x.shape), x))

    # theta = parameters learnt by model
    theta = np.array([0., 0.])

    # number of iterations
    iterations = 0

    # all the iterations of theta
    theta_vals = []
    prev_cost_value = 0.0

    while True:

        # cost function and gradient calculation
        cost_value = 0
        grad = 0
        for i in range(m):
            cost_value += (np.dot(theta, x[i]) - y[i]) ** 2
            grad += (np.dot(theta, x[i]) - y[i]) * x[i]

        # divide by the number of examples etc.
        grad /= m
        cost_value /= (2 * m)

        # append current value of theta to all theta values
        theta_vals.append(np.hstack((theta, np.array(cost_value))))

        # change in the value of theta during gradient update
        diff = eta * grad
        theta = theta - diff
        iterations += 1

        # convergence criterion
        if abs(cost_value - prev_cost_value) < epsilon:
            break

        prev_cost_value = cost_value

    return np.array(theta_vals).T, theta, eta, ('change in cost function is less than ' + str(epsilon)), iterations

def main():

    # read command-line arguments
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    # check for existence of input files
    for c in ['X', 'Y']:
        if not isfile(join(data_dir, 'linear' + c + '.csv')):
            raise Exception('linear' + c + '.csv not found')

    # read from csv file
    x = np.array([np.genfromtxt(join(data_dir, 'linearX.csv'))]).T
    y = np.array([np.genfromtxt(join(data_dir, 'linearY.csv'))]).T

    # normalization step
    x -= np.full_like(x, np.sum(x) / x.shape[0])
    x /= np.sqrt(np.sum(x ** 2) / x.shape[0])
    #y -= np.full_like(y, np.sum(y) / y.shape[0])
    #y /= np.sqrt(np.sum(y ** 2) / y.shape[0])

    # call gradient descent on the given data
    theta_vals, theta, learning_rate, stopping_criteria, total_iterations = grad_desc(x, y)

    # write the output for 1a
    output_file = open(join(out_dir, '1aoutput.txt'), mode='w')

    output_file.write('learning_rate = ' + str(learning_rate) + '\n')
    output_file.write('stopping_criteria = ' + stopping_criteria + '\n')
    output_file.write('theta_0 = ' + str(theta[0]) + '\n')
    output_file.write('theta_1 = ' + str(theta[1]) + '\n')
    output_file.write('total_iterations = ' + str(total_iterations) + '\n')

    output_file.close()

    # PART B: plot the graphs for 1b

    fig1b, ax1b = plt.subplots()
    ax1b.scatter(x, y)
    X0 = np.arange(-2, 5, 0.1)
    ax1b.plot(X0, theta[0] + theta[1] * X0)
    fig1b.savefig(join(out_dir, 'regression_plot.png'))

    # PART C: plot the graph for 1c

    # X, Y, Z - theta_0, theta_1, cost function
    (X, Y), Z = np.meshgrid(np.linspace(-0.5, 2, 1000), np.linspace(-0.7, 0.7, 1000)), 0
    for i in range(x.shape[0]):
        Z += ((X + Y * x[i][0]) - y[i]) ** 2
    Z /= 2 * x.shape[0]

    # actually starting the plot
    fig1c = plt.figure()
    ax1c = fig1c.gca(projection='3d')
    ax1c.plot_surface(X, Y, Z)
    plot1c = ax1c.plot(theta_vals[0, 0], theta_vals[1, 0], theta_vals[2, 0])

    # update function for animation
    def update1c(nums):
        plot1c[0].set_data(theta_vals[0:2, :nums])
        plot1c[0].set_3d_properties(theta_vals[2, :nums])
        return plot1c

    # performing the animation
    anim1c = animation.FuncAnimation(fig1c, update1c, theta_vals.shape[1], interval=200, blit=True)
    update1c(theta_vals.shape[1])
    fig1c.savefig(join(out_dir, '1clast_frame.png'))

    # PART D: plot the graph for 1d

    fig1d, ax1d = plt.subplots()
    ax1d.contour(X, Y, Z, 100)
    plot1d = ax1d.plot(theta_vals[0, 0], theta_vals[1, 0])
    # update function for animation
    def update1d(nums):
        plot1d[0].set_data(theta_vals[0:2, :nums])
        return plot1d
    # performing the animation
    anim1d = animation.FuncAnimation(fig1d, update1d, theta_vals.shape[1], interval=200, blit=True)
    update1d(theta_vals.shape[1])
    fig1d.savefig(join(out_dir, '1dlast_frame.png'))

    # PART E: plot the graphs for 1e

    learning_parameters = [(1e-3, 1e-15), (25e-2, 1e-15), (1e-1, 1e-15)]

    eta, epsilon = learning_parameters[0]
    theta_vals1, _, _, _, _ = grad_desc(x, y, eta, epsilon)

    fig1e1, ax1e1 = plt.subplots()
    ax1e1.contour(X, Y, Z, 100)
    plot1e1 = ax1e1.plot(theta_vals1[0, 0], theta_vals1[1, 0])

    # update function for animation
    def update1e1(nums):
        plot1e1[0].set_data(theta_vals1[0:2, :nums])
        return plot1e1

    # performing the animation
    anim1e1 = animation.FuncAnimation(fig1e1, update1e1, theta_vals1.shape[1], interval=200, blit=True)
    #anim1e1.save(join(out_dir, '1e1.gif'), writer="imagemagick")
    update1e1(theta_vals1.shape[1])
    fig1e1.savefig(join(out_dir, '1e1last_frame.png'))

    eta, epsilon = learning_parameters[1]
    theta_vals2, _, _, _, _ = grad_desc(x, y, eta, epsilon)

    fig1e2, ax1e2 = plt.subplots()
    ax1e2.contour(X, Y, Z, 100)
    plot1e2 = ax1e2.plot(theta_vals2[0, 0], theta_vals2[1, 0])

    # update function for animation
    def update1e2(nums):
        plot1e2[0].set_data(theta_vals2[0:2, :nums])
        return plot1e2

    # performing the animation
    anim1e2 = animation.FuncAnimation(fig1e2, update1e2, theta_vals2.shape[1], interval=200, blit=True)
    #anim1e2.save(join(out_dir, '1e2.gif'), writer="imagemagick")
    update1e2(theta_vals2.shape[1])
    fig1e2.savefig(join(out_dir, '1e2last_frame.png'))

    eta, epsilon = learning_parameters[2]
    theta_vals3, _, _, _, _ = grad_desc(x, y, eta, epsilon)

    fig1e3, ax1e3 = plt.subplots()
    ax1e3.contour(X, Y, Z, 100)
    plot1e3 = ax1e3.plot(theta_vals3[0, 0], theta_vals3[1, 0])

    # update function for animation
    def update1e3(nums):
        plot1e3[0].set_data(theta_vals3[0:2, :nums])
        return plot1e3

    # performing the animation
    anim1e3 = animation.FuncAnimation(fig1e3, update1e3, theta_vals3.shape[1], interval=200, blit=True)
    #anim1e3.save(join(out_dir, '1e3.gif'), writer="imagemagick")
    update1e3(theta_vals3.shape[1])
    fig1e3.savefig(join(out_dir, '1e3last_frame.png'))

    plt.show()

if __name__ == "__main__":
    main()
