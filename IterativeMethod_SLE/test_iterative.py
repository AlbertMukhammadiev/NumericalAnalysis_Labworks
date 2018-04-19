from unittest import TestCase, main
from numpy import array, dot, allclose
from numpy.linalg import norm

from os.path import abspath as os_abspath, join as os_join
lib_path = os_abspath(os_join(__file__, '..', '..', 'GaussianElimination'))
from sys import path as sys_path
sys_path.append(lib_path)

import gaussel as ge
from iterative import *

class IterativeMethodTestCase(TestCase):
    def setUp(self):
        """initial set up"""
        self.A = array(
            [
                [10.161209, 1.2196541, -3.1371353],
                [1.2196541, 7.7162264, 0.72990862],
                [-3.1371353, 0.72990862, 6.1225643]
            ],
            dtype=float
        )

        self.b = array([9.8826956, -6.7155449, -5.7038132], dtype=float)
        self.x0 = zeros(self.A.shape[0])
        self.eps = 1 / 10 **5


    def test_approximate_Jacobi(self):
        """Verify the correctness of the solution of the system"""
        x = approximate_Jacobi(self.A, self.b, self.x0, self.eps)
        b_check = dot(self.A, x)
        self.assertTrue(allclose(self.b, b_check))


    def test_approximate_Nekrasov(self):
        """Verify the correctness of the solution of the system"""
        x = approximate_Nekrasov(self.A, self.b, self.x0, self.eps)
        b_check = dot(self.A, x)
        self.assertTrue(allclose(self.b, b_check))
    

    def test_perform_iteration(self):
        """Verify the Nekrasov method gives a better approximation"""
        x_Jacobi = perform_iteration_Jacobi(self.A, self.b, self.x0)
        x_Nekrasov = perform_iteration_Nekrasov(self.A, self.b, self.x0)
        x = ge.solve_system(self.A, self.b)
        self.assertLess(norm(x - x_Nekrasov), norm(x - x_Jacobi))


    
if __name__ == '__main__':
    main()