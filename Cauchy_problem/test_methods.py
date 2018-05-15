from unittest import TestCase, main
from numpy import allclose
from methods import Runge_Kutta, Adams


class MethodsTestCase(TestCase):
    def setUp(self):
        """initial set up"""
        self.h = 0.1
        self.right_boarder = 1
        self.x0 = 0
        self.y0 = 0
        self.func = '(1 - a * x * x - y * y) / (k - x * y)'

    def test_Adams_vs_RungeKutta(self):
        """Verify the approximations of the solution
        by the methods of the Runge-Kutta and Adams almost coincide
        """

        _, ys1 = Runge_Kutta(self.func, self.x0, self.y0, self.h, 10)
        _, ys2 = Adams(self.func, self.x0, self.y0, self.h, 10)
        print(ys1, '\n', ys2)
        print(type(ys1), type(ys2))
        self.assertTrue(allclose(ys1, ys2, atol=1.e-4))


if __name__ == '__main__':
    main()