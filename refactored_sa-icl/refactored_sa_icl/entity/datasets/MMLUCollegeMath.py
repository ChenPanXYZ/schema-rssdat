import ast
import random
from typing import List

from pathlib import Path
import pandas as pd
import re

from refactored_sa_icl.config.data_paths import MMLU_COLLEGE_MATH_RAW
from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class MMLUCollegeMath(Dataset):
    dataset_name: str
    problems: List[Problem]
    size: int

    def __init__(self, size: 1000000000):
        self.size = size
        self.load_problems()

    '''
    Load Problem from dataset.
    Please refer to the Problem class to see which columns are required.
    '''

    def load_problems(self) -> None:
        """Load problems from the MMLU College Math CSV."""

        self.problems = []

        # Resolve the CSV path relative to the project root, using the
        # centralised constant from ``data_paths``.
        project_root = Path(__file__).resolve().parents[4]
        csv_path = project_root / MMLU_COLLEGE_MATH_RAW
        df = pd.read_csv(csv_path)
        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break

                # correct_answer = str(row['Correct Answer']).replace("\n", "")
                # candidates = [correct_answer, str(row['Incorrect Answer 1']).replace("\n", ""),
                #               str(row['Incorrect Answer 2']).replace("\n", ""),
                #               str(row['Incorrect Answer 3']).replace("\n", "")]
                # random.seed(906)
                # random.shuffle(candidates)
                # label = candidates.index(correct_answer)


                # No need to shuffle.
                fixed = re.sub(r"'\s+'", "', '", row['options'])

                candidates = ast.literal_eval(fixed)



                problem = Problem(
                    id=Dataset.generate_hash(row['question']),
                    question=row['question'],
                    context=None,
                    label=row['answer_index'],
                    candidates=candidates,
                    explanation=None,
                    reference_to=Dataset.generate_hash(row['question']),
                    reference_type="self"
                )
                self.problems.append(problem)
            except Exception as e:
                print(e)
                break

if __name__ == "__main__":
    dataset = MMLUCollegeMath(size=100)
    for problem in dataset.problems:
        print(problem)
        print(problem.get_ground_truth())