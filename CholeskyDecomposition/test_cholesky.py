from unittest import TestCase, main
from numpy import array, real_if_close, allclose

from os.path import abspath as os_abspath, join as os_join
lib_path = os_abspath(os_join(__file__, '..', '..', 'GaussianElimination'))
from sys import path as sys_path
sys_path.append(lib_path)

import gaussel as ge
import cholesky as ch

class CholeskyDecompositionTestCase(TestCase):
    def setUp(self):
        """initial set up"""
        pass


    def test_decomposition(self):
        """Verify the correctness of decomposition"""
        A = array(
            [
                [18, 22, 54, 42],
                [22, 70, 86, 62],
                [54, 86, 174, 134],
                [42, 62, 134, 106]
            ],
            float
        )

        S = ch.decomposition(A)
        A_check = real_if_close(S.T.dot(S))
        self.assertTrue(allclose(A_check, A))
    

    def test_decomposition_im(self):
        """Verify the correctness of decomposition.
        
        The case when matrix have complex elements after decomposition
        """
        
        A = array(
            [
                [2, -2, 3, 0.9],
                [-2, 3, -1, 1],
                [3, -1, 2, 0.5],
                [0.9, 1, 0.5, 2.5]
            ],
            float
        )

        S = ch.decomposition(A)
        A_check = real_if_close(S.T.dot(S))
        self.assertTrue(allclose(A_check, A))

    def test_system_solving(self):
        """Verify the correctness of the solution of the system"""
        A = array(
            [
                [2, -2, 3, 0.9],
                [-2, 3, -1, 1],
                [3, -1, 2, 0.5],
                [0.9, 1, 0.5, 2.5]
            ],
            float
        )

        b = array([-5.00085, 2.76880, 0.89615, 1.35010])
        x = ch.solve_system(A, b)
        self.assertTrue(allclose(A.dot(x), b))


if __name__ == '__main__':
    main()