import unittest
from pathlib import Path

from pyspla import INT, Matrix, Vector
from tests.config import cfg


def read_matrix_from_files(I_path, J_path, V_path, shape):
    I = []
    J = []
    V = []

    with open(I_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                I.append(int(line))

    with open(J_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                J.append(int(line))

    with open(V_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                V.append(int(line))

    return Matrix.from_lists(I, J, V, shape, INT)


def read_matrix_from_file(filepath):
    I = []
    J = []
    V = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                i, j, v = line.split()
                I.append(int(i))
                J.append(int(j))
                V.append(int(v))

    n_rows = max(I) + 1 if I else 3
    n_cols = max(J) + 1 if J else 3

    return Matrix.from_lists(I, J, V, (n_rows, n_cols), INT)


class TestMatrix(unittest.TestCase):

    def test_matrix_creation(self):
        M = Matrix((3, 4), INT)
        self.assertEqual(M.n_rows, 3)
        self.assertEqual(M.n_cols, 4)
        self.assertEqual(M.shape, (3, 4))

    def test_matrix_set_get(self):
        M = Matrix((3, 3), INT)
        M.set(0, 1, 5)
        M.set(1, 2, 10)
        M.set(2, 0, 15)

        self.assertEqual(M.get(0, 1), 5)
        self.assertEqual(M.get(1, 2), 10)
        self.assertEqual(M.get(2, 0), 15)
        self.assertEqual(M.get(1, 1), 0)

    def test_matrix_eadd(self):
        inputs, results, total = cfg.get_test_cases("matrix_eadd", 2)

        for i in range(total):
            with self.subTest(case=i):
                A = read_matrix_from_file(inputs[0][i])
                B = read_matrix_from_file(inputs[1][i])
                expected = read_matrix_from_file(results[i])

                result = A.eadd(INT.PLUS, B)
                self.assertEqual(result.to_lists(), expected.to_lists())

    # def test_matrix_mxm(self):
    #     inputs, results, total = cfg.get_test_cases("matrix_mxm", 2)
    #
    #     for i in range(total):
    #         with self.subTest(case=i):
    #             A = read_matrix_from_file(inputs[0][i])
    #             B = read_matrix_from_file(inputs[1][i])
    #             expected = read_matrix_from_file(results[i])
    #
    #             result = A.mxm(B, INT.MULT, INT.PLUS)
    #             self.assertEqual(result.to_lists(), expected.to_lists())

    def test_matrix_transpose(self):
        inputs, results, total = cfg.get_test_cases("matrix_transpose", 1)

        for i in range(total):
            with self.subTest(case=i):
                M = read_matrix_from_file(inputs[0][i])
                expected = read_matrix_from_file(results[i])

                result = M.transpose()
                self.assertEqual(result.to_lists(), expected.to_lists())

    def test_matrix_reduce_by_row(self):
        inputs, results, total = cfg.get_test_cases("matrix_reduce_by_row", 1)

        for i in range(total):
            with self.subTest(case=i):
                M = read_matrix_from_file(inputs[0][i])

                I_vec, V_vec = [], []
                with open(results[i], 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            idx, val = line.split()
                            I_vec.append(int(idx))
                            V_vec.append(int(val))

                size = max(I_vec) + 1 if I_vec else 3
                expected = Vector.from_lists(I_vec, V_vec, size, INT)

                result = M.reduce_by_row(INT.PLUS)
                self.assertEqual(result.to_lists(), expected.to_lists())