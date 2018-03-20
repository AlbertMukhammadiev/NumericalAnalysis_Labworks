from unittest import TestCase, main
from numpy import array, random, around, count_nonzero, array_equal
from gaussel import *

class GaussianEliminationTestCase(TestCase):
    def setUp(self):
        """initial set up"""
        self.A = array(
            [
                [2, 5, 4, 1],
                [1, 3, 2, 1],
                [2, 10, 9, 7],
                [3, 8, 9, 2]
            ],
            dtype=float
        )

        self.b = array([20, 11, 40, 37])

    def test_forward_elimination(self):
        """Verify the matrix is upper triangular after elimination"""
        matrix = random.rand(100, 100)
        forward_elimination(matrix)
        n = matrix.shape[0]
        for i in range(n):
            self.assertFalse(count_nonzero(matrix[i][:i]))

        # for modified version of forward elimination
        matrix_m = random.rand(100, 100)
        forward_elimination_m(matrix_m)
        n = matrix_m.shape[0]
        for i in range(n):
            self.assertFalse(count_nonzero(matrix_m[i][:i]))

    def test_system_solving(self):
        """Verify the correctness of the solution of the system"""
        x = solve_system(self.A.copy(), self.b, False)
        b_check = self.A.dot(x)
        self.assertTrue(array_equal(self.b, b_check))

        # for modified version of forward elimination
        x = solve_system(self.A.copy(), self.b, True)
        b_check = self.A.dot(x)
        self.assertTrue(array_equal(self.b, b_check))

    def test_back_substitution(self):
        """Verify the correctness of the back substitution"""
        A = array(
            [
                [1, 2, 3, -2, 1],
                [2, -1, -2, -3, 2],
                [3, 2, -1, 2, -5],
                [2, -3, 2, 1, 11]
            ],
            float
        )

        forward_elimination(A)
        x = back_substitution(A)
        true_x = array([2 / 3, -43 / 18, 13 / 9, -7 / 18], float)
        self.assertTrue(array_equal(around(x, 5), around(true_x, 5)))

if __name__ == '__main__':
    main()