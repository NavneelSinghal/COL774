import numpy as np

def computeParametersGaussianKernel(x):
    squares = np.einsum('ij,ij->i', x, x)
    squares = np.hstack(tuple([squares.reshape((squares.shape[0], 1))] * squares.shape[0]))
    squares += squares.T
    squares -= 2 * np.einsum('ik,jk->ij', x, x)
    print(squares)
    pass

def squaredDistanceMatrixOld(x, y):
    squares_x = np.einsum('ij,ij->i', x, x)
    squares_y = np.einsum('ij,ij->i', y, y)
    squares_x_fill = np.vstack(tuple([squares_x] * squares_y.shape[0]))
    return np.vstack(tuple([squares_y] * squares_x.shape[0])) + squares_x_fill.T - 2 * np.matmul(x, y.T)#2 * np.einsum('ik,jk->ij', x, y)

a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
#print('a:')
#print(a)
#print('b:')
#print(b)
#P = squaredDistanceMatrixOld(a, b)
#alpha = np.array([1., 2., 3.])
#y = np.array([-1., -1., 1.])
#print('squared distance matrix:')
#print(P)
#print(alpha)
#print(y)
#print(np.einsum('i,i,ij->j', alpha, y, P))

# a[i, j] = norm of difference of i, jth elements
def squaredDistanceMatrix(x, y, same=False):
    if same:
        squares = np.einsum('ij,ij->i', x, x)
        squares_fill = np.tile(squares, (squares.shape[0], 1))
        return squares_fill + squares_fill.T - 2 * np.matmul(x, x.T) #2 * np.einsum('ik,jk->ij', x, x)
    squares_x = np.einsum('ij,ij->i', x, x)
    squares_y = np.einsum('ij,ij->i', y, y)
    return np.tile(squares_y, (squares_x.shape[0], 1)) + np.tile(squares_x, (squares_y.shape[0], 1)).T - 2 * np.matmul(x, y.T)#2 * np.einsum('ik,jk->ij', x, y)

a = np.random.rand(4500, 1000)
b = np.random.rand(1000, 1000)

import time
t = time.time()
c = squaredDistanceMatrixOld(a, a)
print(time.time() - t)
t = time.time()
d = squaredDistanceMatrix(a, a, True)
print(time.time() - t)
print(np.max(np.abs(c - d)))
