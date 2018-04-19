"""Tool for solving systems of linear equations
using Cholesky decomposition

Functions:
    solve_system(A, b) -> ndarray
    decomposition(matrix) -> ndarray
    print_log() -> None
"""

import numpy as np
import cmath

from os.path import abspath as os_abspath, join as os_join
lib_path = os_abspath(os_join(__file__, '..', '..', 'GaussianElimination'))
from sys import path as sys_path
sys_path.append(lib_path)
import gaussel as ge


def print_log():
    for key, value in _log.items():
        print(key + ':\n', value, end='\n\n')


def solve_system(A, b):
    """Solve system of linear equations A x = b and return x.

    Arguments:
    A -- symmetric matrix
    b -- vector
    """

    _log['symmetric matrix A'] = A.copy()
    _log['vector b'] = b.copy()

    S = decomposition(A)
    _log['upper triangular matrix S'] = S.copy()

    A_check = S.T.dot(S)
    _log['S.T * S'] = A_check

    y = ge.solve_system(S.T, b)
    _log['vector y'] = y.copy()

    x = ge.solve_system(S, y)
    _log['vector x (solution of system)'] = x.copy()

    _log['A * x'] = A.dot(x)

    return x


def decomposition(matrix):
    """Apply Cholesky decomposition to the given matrix.
    
    Return upper triangular matrix.
    Arguments:
    matrix -- symmetric matrix
    """

    n = matrix.shape[0]
    S = np.zeros((n, n), complex)
    for i in range(n):
        a = cmath.sqrt(matrix[i, i] - sum(S[:, i] * S[:, i]))
        S[i, i] = a
        for j in range(i + 1, n):
            S[i, j] = (matrix[i, j] - sum(S[:, i] * np.conj(S[:, j]))) / S[i, i]

    return S


_log = dict()