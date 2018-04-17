import numpy as np
import gaussel as ge
import cmath

_log = dict()


def print_log():
    for key, value in _log.items():
        print(key + ':\n', value, end='\n\n')


def solve_system(A, b):
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
    n = matrix.shape[0]
    S = np.zeros((n, n), complex)
    for i in range(n):
        a = cmath.sqrt(matrix[i, i] - sum(S[:, i] * S[:, i]))
        S[i, i] = a
        for j in range(i + 1, n):
            S[i, j] = (matrix[i, j] - sum(S[:, i] * np.conj(S[:, j]))) / S[i, i]

    return S