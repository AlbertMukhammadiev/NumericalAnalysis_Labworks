"""Tool for finding the maximum modulo eigenvalue
using Power iteration (one of the Eigenvalue algorithm)
and his modification(power iteration with scalar product)

Functions:
    power_iteration(A, eps) -> (float, ndarray)
    power_iteration_m(A, eps) -> (float, ndarray)
"""

from numpy import random, dot
from numpy.linalg import norm

def power_iteration_m(A, eps):
    """Find the maximum modulo eigenvalue with a given accuracy.

    This function applies a scalar product to accelerate convergence.

    Arguments:
    A -- square matrix
    eps -- accuracy
    """

    vector = random.rand(A.shape[0])
    value = random.rand()
    diff = 2 * eps
    while eps < diff:
        next_vector1 = dot(A, vector)
        next_vector2 = dot(A.T, vector)
        next_value = dot(next_vector1, next_vector2) / dot(vector, next_vector2)
        diff = abs(next_value - value)
        
        value = next_value
        vector = next_vector1

    return value, vector / norm(vector, ord=2)

def power_iteration(A, eps):
    """Find the maximum modulo eigenvalue with a given accuracy.
    
    Arguments:
    A -- square matrix
    eps -- accuracy
    """
    vector = random.rand(A.shape[0])
    value = random.rand()
    diff = 2 * eps
    while eps < diff:
        next_vector = dot(A, vector)
        next_value = next_vector[0] / vector[0]
        diff = abs(next_value - value)

        value = next_value
        vector = next_vector

    return value, vector / norm(vector, ord=2)