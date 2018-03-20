"""Tool for solving systems of linear equations
using Gaussian elimination algorithm

Functions:
    solve_system(A, b, is_modified=False) -> ndarray
    forward_elimination(matrix) -> None
    forward_elimination_m(matrix) -> None
    back_substitution(matrix) -> ndarray
    print_log() -> None
"""

__all__ = [
    'solve_system', 'forward_elimination_m', 'forward_elimination',
    'back_substitution', 'print_log'
]

import numpy as np


def solve_system(A, b, is_modified=False):
    """Solve system of linear equations A x = b and return x.

    Arguments:
    A -- square matrix
    b -- vector
    Keyword arguments:
    is_modified -- selection flag of forward elimination(default False)
    """

    log['A'] = A
    log['b'] = b
    log['forward elimination is modified'] = is_modified
    extd_matrix = np.hstack((A, np.array([b]).T))
    log['extended matrix'] = extd_matrix.copy()
    if is_modified:
        forward_elimination_m(extd_matrix)
    else:
        forward_elimination(extd_matrix)

    x = back_substitution(extd_matrix)
    log['triangular matrix'] = extd_matrix
    log['solution of system'] = x
    return x


def forward_elimination_m(matrix):
    """Perform the forward elimination over the given matrix.

    This function applies a forward elimination using the selection
    of the main elements 

    Arguments:
    matrix -- two-dimensional matrix
    """

    ctrl_vector = matrix.sum(axis = 1)
    log['control vector before forward elimination'] = ctrl_vector.copy()
    n, m = matrix.shape[0], matrix.shape[1]
    for k in range(n):
        # search for the main elements(max by column) and rows swapping
        p = k + abs(matrix[k:,k]).argmax()
        matrix[p], matrix[k] = matrix[k], matrix[p].copy()
        ctrl_vector[p], ctrl_vector[k] = ctrl_vector[k], ctrl_vector[p]
        
        ctrl_vector[k] /= matrix[k][k]
        matrix[k] /= matrix[k][k]
        for i in range(k + 1, n):
            ctrl_vector[i] -= ctrl_vector[k] * matrix[i][k]
            matrix[i] -= matrix[k] * matrix[i][k]

    log['control vector after forward elimination'] = ctrl_vector


def forward_elimination(matrix):
    """Perform the forward elimination over the given matrix.

    Arguments:
    matrix -- two-dimensional matrix
    """

    ctrl_vector = matrix.sum(axis = 1)
    log['control vector before forward elimination'] = ctrl_vector.copy()
    n, m = matrix.shape[0], matrix.shape[1]
    for k in range(n):
        ctrl_vector[k] /= matrix[k][k]
        matrix[k] /= matrix[k][k]
        for i in range(k + 1, n):
            ctrl_vector[i] -= ctrl_vector[k] * matrix[i][k]
            matrix[i] -= matrix[k] * matrix[i][k]

    log['control vector after forward elimination'] = ctrl_vector


def back_substitution(matrix):
    """Perform back substitution and return solution of system

    Arguments:
    matrix -- two-dimensional matrix
    
    Attention: given argument must be a upper triangular matrix
    with ones on main diagonal and have n rows and n + 1 columns
    (last column is vector b)
    """

    n = matrix.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i, n] - sum(matrix[i, :n] * x)
    
    return x


def print_log():
    """Print a log that stores step-by-step actions"""
    for key, value in log.items():
        print(key + ':\n', value, end='\n\n')


log = dict()