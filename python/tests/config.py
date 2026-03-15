"""
Configuration for loading test cases.
"""

import os
import pathlib

__all__ = ["Config", "cfg"]


class Config:
    __slots__ = ["data_path"]

    def __init__(self, data_path):
        self.data_path = data_path

    def get_test_cases(self, operation: str, num_inputs: int):
        op_path = self.data_path / operation

        if not op_path.exists():
            return [], [], 0

        cases = sorted([d for d in op_path.iterdir() if d.is_dir()])

        inputs = [[] for _ in range(num_inputs)]
        results = []

        for case in cases:
            for i in range(num_inputs):
                input_file = case / f"input_{i}.txt"
                if input_file.exists():
                    inputs[i].append(str(input_file))
                else:
                    inputs[i].append(None)

            result_file = case / "result.txt"
            results.append(str(result_file) if result_file.exists() else None)

        return inputs, results, len(cases)


PATH = pathlib.Path(__file__)
DATA_PATH = PATH.parent / "data"
cfg = Config(DATA_PATH)