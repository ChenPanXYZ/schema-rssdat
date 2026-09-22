import random
from typing import List

import pandas as pd

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class Math(Dataset):
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

    def load_problems(self):
        # load csv gpqa_main.csv
        self.problems = []
        # get path of this file
        import os
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        df = pd.read_csv(f'{path}/raw_files/math.csv')
        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break

                correct_answer = str(row['answer'])
                random.seed(906)

                problem = Problem(
                    id=Dataset.generate_hash(row['problem']),
                    question=row['problem'],
                    context=None,
                    label=correct_answer,
                    candidates=None,
                    explanation=row['solution'],
                    reference_to=Dataset.generate_hash(row['problem']),
                    reference_type="self"
                )
                self.problems.append(problem)
            except Exception as e:
                print("Error loading problem:", e)
                break


if __name__ == "__main__":
    dataset = Math(size=10000)
    for problem in dataset.problems:
        print(problem.get_ground_truth())
    print(len(dataset.problems))
