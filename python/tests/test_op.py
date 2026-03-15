import unittest
from pathlib import Path

from pyspla import INT, Vector, Scalar
from tests.config import cfg


def read_vector_from_file(filepath):
    indices = []
    values = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                idx, val = line.split()
                indices.append(int(idx))
                values.append(int(val))

    size = max(indices) + 1 if indices else 5
    return Vector.from_lists(indices, values, size, INT)


class TestOperations(unittest.TestCase):

    def test_unary_ops(self):
        base_path = Path(__file__).parent / "data" / "op_unary"
        cases = sorted([d for d in base_path.iterdir() if d.is_dir()])

        for i, case in enumerate(cases):
            input_file = case / "input.txt"
            abs_file = case / "result_abs.txt"
            ainv_file = case / "result_ainv.txt"

            v = read_vector_from_file(input_file)

            with self.subTest(case=i, op="abs"):
                expected = read_vector_from_file(abs_file)
                result = v.map(INT.ABS)
                self.assertEqual(result.to_lists(), expected.to_lists())

            with self.subTest(case=i, op="ainv"):
                expected = read_vector_from_file(ainv_file)
                result = v.map(INT.AINV)
                self.assertEqual(result.to_lists(), expected.to_lists())

    def test_binary_ops(self):
        base_path = Path(__file__).parent / "data" / "op_binary"
        cases = sorted([d for d in base_path.iterdir() if d.is_dir()])

        for i, case in enumerate(cases):
            a_file = case / "input_0.txt"
            b_file = case / "input_1.txt"
            plus_file = case / "result_plus.txt"
            mult_file = case / "result_mult.txt"
            max_file = case / "result_max.txt"

            a = read_vector_from_file(a_file)
            b = read_vector_from_file(b_file)

            with self.subTest(case=i, op="plus"):
                expected = read_vector_from_file(plus_file)
                result = a.eadd(INT.PLUS, b)
                for idx in range(expected.n_rows):
                    self.assertEqual(result.get(idx), expected.get(idx))

            with self.subTest(case=i, op="mult"):
                expected = read_vector_from_file(mult_file)
                result = a.emult(INT.MULT, b)
                for idx in range(expected.n_rows):
                    self.assertEqual(result.get(idx), expected.get(idx))

            with self.subTest(case=i, op="max"):
                expected = read_vector_from_file(max_file)
                result = a.eadd(INT.MAX, b)
                for idx in range(expected.n_rows):
                    self.assertEqual(result.get(idx), expected.get(idx))
