from unittest import TestCase, main
from sympy import sympify, symbols
from numpy import array, allclose
from tridiagonal import solve_diffeq


class TridiagonalTestCase(TestCase):
    def setUp(self):
        """initial set up"""

    def test_model_solution(self):
        """Verify the resulting approximation is correct."""
        x = symbols('x')
        # yx = sympify('2 * x**3 - 3 * x + 4')
        yx = sympify('2 * sin(x) - 3 * x + 4')
        dyx = yx.diff(x)
        d2yx = dyx.diff(x)

        h = 1.e-2
        px, qx = sympify('1'), sympify('-1')
        alpha1, alpha2, beta1, beta2 = -1, 1, 1, 1
        a, b = 0, 1

        fx = d2yx + px * dyx + qx * yx
        A = alpha1 * yx.subs({x: a}) + alpha2 * dyx.subs({x: a})
        B = beta1 * yx.subs({x: b}) + beta2 * dyx.subs({x: b})
        xs, ys = solve_diffeq(px, qx, fx, alpha1, alpha2, A, beta1, beta2, B, h, a, b)
        exact_ys = array([yx.subs({x: x_}) for x_ in xs], dtype=float)
        self.assertTrue(allclose(ys, exact_ys))


if __name__ == '__main__':
    main()