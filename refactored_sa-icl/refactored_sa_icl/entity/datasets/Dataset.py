from typing import List

import hashlib

from refactored_sa_icl.entity.problems.Problem import Problem


class Dataset:
    dataset_name: str
    problems: List[Problem]
    size: int

    def __init__(self, size: int) -> None:
        self.size = size
        self.load_problems()

    """
    Load Problem from dataset.
    Please refer to the Problem class to see which columns are required.
    """

    def load_problems(self) -> List[Problem]:
        raise NotImplementedError

    @staticmethod
    def generate_hash(input_string: str) -> str:
        """
        Generate a 36-character alphanumeric hash from the input string.
        """
        sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
        return sha256_hash[:36]
