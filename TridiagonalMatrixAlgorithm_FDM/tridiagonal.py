"""Tool for solving the ordinary differential equation of order 2
by the tridiagonal matrix algorithm and finite difference method.

Functions:
    solve_TDMA(below, main, above, vector) -> ndarray
    solve_diffeq(px, qx, fx, alpha1, alpha2, A,
                 beta1, beta2, B, h, a, b) -> tuple(ndarray, ndarray)
"""
from numpy import zeros, array
from sympy import sympify, symbols


def solve_diffeq(px, qx, fx, alpha1, alpha2, A, beta1, beta2, B, h, a, b):
    """solve differential equation of 2nd order:
    
    Arguments from the equation below:
    y'' + p(x) y' + q(x) y = f(x), x in [a, b]
    alpha1 * y(a) + alpha2 * y'(a) = A
    beta1 * y(b) + beta2 * y'(b) = B
    """
    x = symbols('x')
    px, qx, fx = sympify(px), sympify(qx), sympify(fx)
    if (alpha1 ** 2 + alpha2 ** 2 == 0 or
        beta1 ** 2 + beta2 ** 2 == 0 or
        alpha1 * alpha2 > 0 or
        beta1 * beta2 < 0):
        print('the algorithm is unstable for these boundary conditions')
    
    n = int((b - a) / h)
    ac, bc, cc, gc = zeros(n), zeros(n + 1), zeros(n), zeros(n + 1)
    for i in range(1, n):
        ac[i] = 1 + h / 2 * px.subs({x: a + i * h})
        bc[i] = - 2 + h ** 2 * qx.subs({x: a + i * h})
        cc[i - 1] = 1 - h / 2 * px.subs({x: a + i * h})
        gc[i] = h ** 2 * fx.subs({x: a + i * h})

    fraction = (cc[0] * alpha2 + 2 * h * ac[1] * alpha1 - 3 * alpha2 * ac[1])
    kappa1 = alpha2 * (-bc[1] - 4 * ac[1]) / fraction
    nu1 = (2 * h * ac[1] * A + alpha2 * gc[1]) / fraction

    fraction = beta2 * (ac[n - 1] - cc[n - 2] * (beta1 * 2 * h + 3 * beta2))
    kappa2 = beta2 * (-bc[n - 1] - 4 * cc[n - 2]) / fraction
    nu2 = (gc[n - 1] * beta2 - 2 * h * B * cc[n - 2]) / fraction

    ac[0] = - kappa1
    cc[n - 1] = - kappa2
    bc[0], bc[n] = 1, 1
    gc[0], gc[n] = nu1, nu2
    ys = solve_TDMA(cc, bc, ac, gc)    
    return array([a + i * h for i in range(n + 1)]), ys


def solve_TDMA(below, main, above, vector):
    '''solve SoLE Ax = b by Tri Diagonal Matrix Algorithm

    Arguments:
    below -- the first diagonal below the main diagonal
    main -- main diagonal
    above --  the first diagonal above the main diagonal
    vector -- vector b
    '''
    n_equations = len(vector)
    for i in range(1, n_equations):
        mc = below[i - 1] / main[i - 1]
        main[i] = main[i] - mc * above[i - 1]
        vector[i] = vector[i] - mc * vector[i - 1]

    xs = main
    xs[-1] = vector[-1] / main[-1]
    for i in range(n_equations - 2, -1, -1):
        xs[i] = (vector[i] - above[i] * xs[i + 1]) / main[i]

    return xs


if __name__ == '__main__':
    px = 'sin(x) / (1 + x**2)**0.5'
    qx = '-(1 + x + x * cos(x ** 2))'
    fx = '1'
    alpha1, alpha2, A = -0.1, 1, 0.4
    beta1, beta2, B = 0.2, 1, 0.5
    a, b = 0, 1

    h = 0.1
    xs1, ys1 = solve_diffeq(px, qx, fx, alpha1, alpha2, A,
                                beta1, beta2, B, h, a, b)

    h = 0.01
    xs01, ys01 = solve_diffeq(px, qx, fx, alpha1, alpha2, A,
                                beta1, beta2, B, h, a, b)

    h = 0.001
    xs001, ys001 = solve_diffeq(px, qx, fx, alpha1, alpha2, A,
                                beta1, beta2, B, h, a, b)

    print('\th = 0.1\t\t\t\t', 'h = 0.01\t\t\t', 'h = 0.001')
    for i in range(11):
        print(ys1[i], '\t\t', ys01[i * 10], '\t\t', ys001[i * 100])