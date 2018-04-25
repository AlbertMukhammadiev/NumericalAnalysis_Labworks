from unittest import TestCase, main
from numpy import array, allclose

from os.path import abspath as os_abspath, join as os_join
lib_path = os_abspath(os_join(__file__, '..', '..', 'PowerIteration_EigenvalueAlgorithm'))
from sys import path as sys_path
sys_path.append(lib_path)

from powiter import power_iteration_m
from jacobi import calculate_eigenvalues, calculate_eigenvalues_m

class JacobiEigenvalueTestCase(TestCase):
    def setUp(self):
        """initial set up"""
        self.eps = 10 ** -10
        self.A = array(
            [
                [1.42, 7.45, 0.38],
                [7.45, 1.61, 0.56],
                [0.38, 0.56, 0.82]
            ]
        )

    def test_calculate_eigenvalues(self):
        """Verify the found eigenvalues contains max modulo eigenvalue"""
        eigenvalues = calculate_eigenvalues(self.A, self.eps)
        eigenvalue, eigenvector = power_iteration_m(self.A, self.eps)
        ndigits = 5
        self.assertIn(eigenvalue.round(ndigits), eigenvalues.round(ndigits))


    def test_calculate_eigenvalues_m(self):
        """Verify the found eigenvalues contains max modulo eigenvalue"""
        eigenvalues = calculate_eigenvalues_m(self.A, self.eps)
        eigenvalue, eigenvector = power_iteration_m(self.A, self.eps)
        ndigits = 5
        self.assertIn(eigenvalue.round(ndigits), eigenvalues.round(ndigits))

if __name__ == '__main__':
    main()