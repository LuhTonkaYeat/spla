import unittest
from pathlib import Path

from pyspla import INT, FLOAT, Scalar
from tests.config import cfg


def read_scalar_from_file(filepath):
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        if '.' in line:
            return float(line)
        else:
            return int(line)


class TestScalar(unittest.TestCase):

    def test_scalar_creation(self):
        s = Scalar(INT, 10)
        self.assertEqual(s.get(), 10)

        s = Scalar(FLOAT, 3.14)
        self.assertAlmostEqual(s.get(), 3.14, places=5)

    def test_scalar_set_get(self):
        s = Scalar(INT)
        s.set(5)
        self.assertEqual(s.get(), 5)

        s.set(42)
        self.assertEqual(s.get(), 42)

    def test_scalar_operations(self):
        base_path = Path(__file__).parent / "data" / "scalar_operations"
        cases = sorted([d for d in base_path.iterdir() if d.is_dir()])

        for i, case in enumerate(cases):
            a_file = case / "input_0.txt"
            b_file = case / "input_1.txt"
            plus_file = case / "result_plus.txt"
            sub_file = case / "result_sub.txt"
            mul_file = case / "result_mult.txt"
            div_file = case / "result_div.txt"

            a_val = read_scalar_from_file(a_file)
            b_val = read_scalar_from_file(b_file)
            a = Scalar(INT, a_val)
            b = Scalar(INT, b_val)

            with self.subTest(case=i, op="add"):
                expected = read_scalar_from_file(plus_file)
                self.assertEqual((a + b).get(), expected)

            with self.subTest(case=i, op="sub"):
                expected = read_scalar_from_file(sub_file)
                self.assertEqual((a - b).get(), expected)

            with self.subTest(case=i, op="mul"):
                expected = read_scalar_from_file(mul_file)
                self.assertEqual((a * b).get(), expected)

            with self.subTest(case=i, op="div"):
                expected = read_scalar_from_file(div_file)
                self.assertEqual((a // b).get(), expected)