"""Tool for solving systems of linear equations
using iterative methods(Jacobi and Nekrasov methods).
Iterative method is a mathematical procedure
that uses an initial guess to generate a sequence
of improving approximate solutions for a class of problems,
in which the n-th approximation is derived from the previous ones.

Functions:
    perform_iteration_Nekrasov(A, b, x0) -> float
    perform_iteration_Jacobi(A, b, x0) -> float
    approximate_Nekrasov(A, b, x0, eps) -> float
    approximate_Jacobi(A, b, x0, eps) -> float
"""

from numpy import zeros


def perform_iteration_Nekrasov(A, b, x0):
    """Returns the next approximating value using the Nekrasov method.

    Arguments:
    A -- square matrix with diagonal predominance
    b -- vector
    x0 -- previous approximation
    """

    n = A.shape[0]
    x = zeros(n)
    x0 = x0.copy()
    for k in range(n):
        product_k = x0 * A[k]
        product_k[k] = 0
        x[k] = (b[k] - sum(product_k)) / A[k, k]
        x0[k] = x[k]

    return x


def perform_iteration_Jacobi(A, b, x0):
    """Returns the next approximating value using the Jacobi method.

    Arguments:
    A -- square matrix with diagonal predominance
    b -- vector
    x0 -- previous approximation
    """

    n = A.shape[0]
    x = zeros(n)
    for k in range(n):
        product_k = x0 * A[k]
        product_k[k] = 0
        x[k] = (b[k] - sum(product_k)) / A[k, k]

    return x


def approximate_Nekrasov(A, b, x0, eps):
    """Find the solution of SLE with a given accuracy
    using Nekrasov method.

    Arguments:
    A -- square matrix with diagonal predominance
    b -- vector
    x0 -- previous approximation
    eps -- accuracy
    """

    while True:
        prev = x0
        _log['Nekrasov'].append(prev)
        x0 = perform_iteration_Nekrasov(A, b, prev)
        if (sum(abs(x0 - prev))) < eps:
            break

    return x0


def approximate_Jacobi(A, b, x0, eps):
    """Find the solution of SLE with a given accuracy
    using Nekrasov method.

    Arguments:
    A -- square matrix with diagonal predominance
    b -- vector
    x0 -- previous approximation
    eps -- accuracy
    """

    while True:
        prev = x0
        _log['Jacobi'].append(prev)
        x0 = perform_iteration_Jacobi(A, b, prev)
        if (sum(abs(x0 - prev))) < eps:
            break

    return x0


_log = {'Nekrasov': [], 'Jacobi': []}