from unittest import TestCase, main
from numpy import array, allclose
from powiter import *

class PowerIterationTestCase(TestCase):
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

    def test_power_iteration(self):
        """Verify the found eigenvector and eigenvalue are correct"""
        eigenvalue, eigenvector = power_iteration(self.A, self.eps)
        self.assertTrue(
            allclose(self.A.dot(eigenvector), eigenvalue * eigenvector)
        )

    def test_power_iteration_m(self):
        """Verify the found eigenvector and eigenvalue are correct"""
        eigenvalue, eigenvector = power_iteration_m(self.A, self.eps)
        self.assertTrue(
            allclose(self.A.dot(eigenvector), eigenvalue * eigenvector, )
        )

if __name__ == '__main__':
    main()