import sympy as sp
from matplotlib.pyplot import plot, show
from numpy import array, hstack, exp


#TODO finite differences and Adams


def supremum_abs(func, xs, ys):
    supremum = 0
    x, y = sp.symbols('x, y')
    for x_, y_ in zip(xs, ys):
        value = abs(func.subs({x: x_, y: y_}))
        if value > supremum:
            supremum = value

    return supremum


def finite_differences(func, a, b, n, m):
    dx = (b - a) / (n - 1);

    xs = array(x + i * h for i in range(n))
    ys = array(func(x) for x in xs)
    plus = zeros((n, n))
    ar = hstack((ys, plus))
    for j in range(1, m + 1):
        for i in range(n - j + 1):
            ar[i, j] = ar[i + 1, j - 1] - ar[i, j - 1];

    print(ar)



def Runge_Kutta(func, x0, y0, h, nsteps):
    x, y, k, a = sp.symbols('x, y, k, a')
    func = sp.sympify(func).subs({k: 1, a: 1})
    
    xs, ys = [x0], [y0]
    for k in range(nsteps):
        k1 = h * func.subs({x: xs[k], y: ys[k]})
        k2 = h * func.subs({x: xs[k] + h / 2, y: ys[k] + k1 / 2})
        k3 = h * func.subs({x: xs[k] + h / 2, y: ys[k] + k2 / 2})
        k4 = h * func.subs({x: xs[k] + h, y: ys[k] + k3})

        xs.append(xs[k] + h)
        ys.append(ys[k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    return xs, ys


def forward_Euler(func, h, x0, y0, right):
    x, y, k, a = sp.symbols('x, y, k, a')
    func = sp.sympify(func).subs({k: 1, a: 1})

    nsteps = int((right - x0) / h)
    xs, ys = [x0], [y0]
    for k in range(nsteps):
        xs.append(xs[k] + h)
        ys.append(ys[k] + h * func.subs({x: xs[k], y: ys[k]}))

    M1 = supremum_abs(func, xs, ys)
    M2 = supremum_abs(func.diff(x), xs, ys)
    M3 = supremum_abs(func.diff(y), xs, ys)
    M4 = M2 + M1 * M3
    error = M4 / M3 * h * exp(float(M3 * (xs[k] - xs[0])))

    return xs, ys, error


if __name__ == '__main__':
    h = 0.1
    right = 1
    x0, y0 = 0, 0
    fun = '(1 - a * x * x - y * y) / (k - x * y)'
    xs1, ys1, err1 = forward_Euler(fun, h, x0, y0, right)
    xs2, ys2 = Runge_Kutta(fun, x0, y0, h, 10)
    finite_differences(fun,0 , 1, 10, 7)
    # plot(xs1, ys1, xs2, ys2)
    # show()