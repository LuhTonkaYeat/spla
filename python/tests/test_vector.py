import unittest
from pathlib import Path

from pyspla import INT, Vector
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


class TestVectorAdd(unittest.TestCase):

    def test_eadd(self):
        inputs, results, total = cfg.get_test_cases("vector_add", 2)

        for i in range(total):
            with self.subTest(case=i):
                a = read_vector_from_file(inputs[0][i])
                b = read_vector_from_file(inputs[1][i])
                expected = read_vector_from_file(results[i])

                result = a.eadd(INT.PLUS, b)
                self.assertEqual(result.to_lists(), expected.to_lists())


class TestVectorFromList(unittest.TestCase):

    def test_from_list(self):
        base_path = Path(__file__).parent / "data" / "vector_from_list"
        cases = sorted([d for d in base_path.iterdir() if d.is_dir()])

        for i, case in enumerate(cases):
            input_file = case / "input.txt"
            result_file = case / "result.txt"

            indices = []
            values = []
            with open(input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        idx, val = line.split()
                        indices.append(int(idx))
                        values.append(int(val))

            size = max(indices) + 1
            v = Vector.from_lists(indices, values, size, INT)
            expected = read_vector_from_file(result_file)

            for idx in range(expected.n_rows):
                self.assertEqual(v.get(idx), expected.get(idx))