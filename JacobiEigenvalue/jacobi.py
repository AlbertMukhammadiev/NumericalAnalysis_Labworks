"""Tool for finding the eigenvalues using Jacobi eigenvalue algorithm

Functions:
    calculate_eigenvalues(A, eps) -> ndarray
"""

from numpy import diag, diag_indices, argmax, unravel_index, ones, sign, dot


def calculate_eigenvalues(A, eps):
    """Find the eigenvalues of matrix with a given accuracy.
    
    Arguments:
    A -- symmetric matrix
    eps -- accuracy
    """

    n = A.shape[0]
    di = diag_indices(n)
    zeroed_diag_A = A.copy()
    zeroed_diag_A[di] = 0
    while True:
        indices = unravel_index(argmax(abs(zeroed_diag_A), axis=None), zeroed_diag_A.shape)
        i, j = min(indices), max(indices)
        
        T_ij = diag(ones(n))
        d = ((A[i, i] - A[j, j]) ** 2 + 4 * A[i,j] ** 2) ** 0.5
        T_ij[i, i] = (((1 + abs(A[i, i] - A[j, j]) / d)) / 2) ** 0.5
        T_ij[j, j] = T_ij[i, i]
        T_ij[j, i] = sign(A[i, j] * (A[i, i] - A[j, j])) * (((1 - abs(A[i, i] - A[j, j]) / d)) / 2) ** 0.5
        T_ij[i, j] = - T_ij[j, i]

        A = dot(dot(T_ij.T, A), T_ij)
        
        zeroed_diag_A = A.copy()
        zeroed_diag_A[di] = 0
        if sum(sum(zeroed_diag_A ** 2)) < eps:
            break

    return A[di]


def calculate_eigenvalues_m(A, eps):
    """Find the eigenvalues of matrix with a given accuracy.

    This function just uses formulas without using matrix multiplication
    
    Arguments:
    A -- symmetric matrix
    eps -- accuracy
    """
    
    n = A.shape[0]
    di = diag_indices(n)
    zeroed_diag_A = A.copy()
    zeroed_diag_A[di] = 0
    while True:
        indices = unravel_index(argmax(abs(zeroed_diag_A), axis=None), zeroed_diag_A.shape)
        i, j = min(indices), max(indices)

        d = ((A[i, i] - A[j, j]) ** 2 + 4 * A[i,j] ** 2) ** 0.5
        cosfi = (((1 + abs(A[i, i] - A[j, j]) / d)) / 2) ** 0.5
        sinfi = sign(A[i, j] * (A[i, i] - A[j, j])) * (((1 - abs(A[i, i] - A[j, j]) / d)) / 2) ** 0.5

        C = A.copy()
        C[i, i] = (A[i, i] + A[j, j]) / 2 + sign(A[i, i] - A[j, j]) * d / 2
        C[j, j] = (A[i, i] + A[j, j]) / 2 - sign(A[i, i] - A[j, j]) * d / 2
        C[i, j], C[j, i] = 0, 0
        for k in range(n):
            if k == i or k == j:
                continue

            C[i, k] = cosfi * A[k, i] + sinfi * A[k, j]
            C[k, i] = C[i, k]
            C[j, k] = - sinfi * A[k, i] + cosfi * A[k, j]
            C[k, j] = C[j, k]
        
        A = C
        zeroed_diag_A = A.copy()
        zeroed_diag_A[di] = 0
        if sum(sum(zeroed_diag_A ** 2)) < eps:
            break

    return A[di]