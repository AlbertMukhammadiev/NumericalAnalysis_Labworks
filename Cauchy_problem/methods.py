"""Tool for solving the first-order ODE(y' = f(x, y))
by various algorithms.

Functions:
    finite_differences(ys, m) -> ndarray
    supremum_abs(func, min_x, max_x, min_y, max_y) -> float
    Runge_Kutta(func, x0, y0, h, nsteps) -> tuple(ndarray, ndarray)
    Euler(func, x0, y0, h, nsteps) -> tuple(ndarray, ndarray, float)
    Adams(func, x0, y0, h, nsteps) -> tuple(ndarray, ndarray)
"""
from sympy import sympify, symbols
from numpy import zeros, array, exp, hstack
from matplotlib.pyplot import plot, legend, show


def finite_differences(ys, m):
    """Return finite differences to a given order.
    
    Arguments:
    ys -- function values
    m -- order of calculated differences
    """

    n = len(ys)
    result = zeros((n, n))
    result[:, 0] = ys
    for j in range(min(n - 1, m)):
        for i in range(n - j - 1):
            result[i, j + 1] = result[i + 1, j] - result[i, j]

    return result


def supremum_abs(func, min_x, max_x, min_y, max_y):
    """Return the supremum module in the domain of definition.
    
    Arguments:
    func -- sympy function of x and y
    (min_x, max_x) -- x-axis segment
    (min_y, max_y) -- y-axis segment
    """
    supremum = 0
    x, y = symbols('x, y')
    dx = (max_x - min_x) / 10
    dy = (max_y - min_y) / 10
    y_ = min_y
    for _ in range(10):
        x_ = min_x
        for _ in range(10):
            x_ += dx
            value = abs(func.subs({x: x_, y: y_}))
            if value > supremum:
                supremum = value

        y_ += dy

    return supremum


def Runge_Kutta(func, x0, y0, h, nsteps):
    """Solve the first-order ODE(y'=f(x, y)) by the Runge-Kutta method.

    Arguments:
    func -- f(x, y) in string representation
    x0, y0 -- initial value
    h -- step size
    nsteps -- number of steps
    """
    x, y, k, a = symbols('x, y, k, a')
    func = sympify(func).subs({k: 3, a: 3})

    xs, ys = zeros(nsteps + 1), zeros(nsteps + 1)
    xs[0], ys[0] = x0, y0
    for k in range(nsteps):
        k1 = h * func.subs({x: xs[k], y: ys[k]})
        k2 = h * func.subs({x: xs[k] + h / 2, y: ys[k] + k1 / 2})
        k3 = h * func.subs({x: xs[k] + h / 2, y: ys[k] + k2 / 2})
        k4 = h * func.subs({x: xs[k] + h, y: ys[k] + k3})

        xs[k + 1] = xs[k] + h
        ys[k + 1] = ys[k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return xs, ys


def Euler(func, x0, y0, h, nsteps):
    """Solve the first-order ODE(y' = f(x, y)) by the Euler method.

    Arguments:
    func -- f(x, y) in string representation
    x0, y0 -- initial value
    h -- step size
    nsteps -- number of steps
    """
    x, y, k, a = symbols('x, y, k, a')
    func = sympify(func).subs({k: 3, a: 3})

    xs, ys = zeros(nsteps + 1), zeros(nsteps + 1)
    xs[0], ys[0] = x0, y0
    for k in range(nsteps):
        xs[k + 1] = xs[k] + h
        ys[k + 1] = ys[k] + h * func.subs({x: xs[k], y: ys[k]})

    M1 = supremum_abs(func, min(xs), max(xs), min(ys), max(ys))
    M2 = supremum_abs(func.diff(x), min(xs), max(xs), min(ys), max(ys))
    M3 = supremum_abs(func.diff(y), min(xs), max(xs), min(ys), max(ys))
    M4 = M2 + M1 * M3
    error = M4 / M3 * h * exp(float(M3 * (xs[k] - xs[0])))

    return xs, ys, error


def Adams(func, x0, y0, h, nsteps):
    """Solve the first-order ODE(y' = f(x, y)) by the Adams method.

    Return ndarrays of points xs, 
    Arguments:
    func -- f(x, y) in string representation
    x0, y0 -- initial value
    h -- step size
    nsteps -- number of steps
    """
    x, y, k, a = symbols('x, y, k, a')
    func = sympify(func).subs({k: 3, a: 3})

    xs, ys = Runge_Kutta(func, x0, y0, h, 4)
    diff_array = finite_differences(array([func.subs({x: x_, y: y_}) * h for x_, y_ in zip(xs, ys)]), 4)
    differences = array([diff_array[4 - i, i] for i in range(5)])
    multipliers = array([1. , 1 / 2, 5 / 12, 3 / 8, 251 / 720])

    xs, ys = hstack((xs, zeros(nsteps - 4))), hstack((ys, zeros(nsteps - 4)))
    for k in range(4, nsteps):
        xs[k + 1] = xs[k] + h
        ys[k + 1] = ys[k] + sum(differences * multipliers)
        
        temp = differences.copy()
        differences[0] = h * func.subs({x: xs[k + 1], y: ys[k + 1]})
        for i in range(4):
            differences[i + 1] = differences[i] - temp[i]

    return xs, ys


if __name__ == '__main__':
    h = 0.1
    right_boarder = 1
    x0 = 0
    y0 = 0
    func = '(1 - a * x * x - y * y) / (k - x * y)'
    nsteps = 10
    xs1, ys1, err1 = Euler(func, x0, y0, h, nsteps)
    xs2, ys2, err2 = Euler(func, x0, y0, h / 2, nsteps * 2)
    xs3, ys3, err3 = Euler(func, x0, y0, h * 2, nsteps // 2)
    xs4, ys4 = Adams(func, x0, y0, h, nsteps)
    xs5, ys5 = Runge_Kutta(func, x0, y0, h, nsteps)

    line1, = plot(xs1, ys1, label='Euler h')
    line2, = plot(xs2, ys2, label='Euler h / 2')
    line3, = plot(xs3, ys3, label='Euler 2 h')
    line4, = plot(xs4, ys4, label='Adams')
    line5, = plot(xs5, ys5, label='Ringe Kutta')
    legend(handles=[line1, line2, line3, line4, line5])
    show()

    print(err1, err2, err3)