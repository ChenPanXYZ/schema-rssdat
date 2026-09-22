from typing import List
import random
import pandas as pd

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class SyntheticBiology(Dataset):
    dataset_name: str
    problems: List[Problem]
    size: int
    def __init__(self, size):
        self.size = size
        self.load_problems()
    '''
    Load Problem from dataset.
    Please refer to the Problem class to see which columns are required.
    '''
    def load_problems(self):
        # load csv synthetic_data.csv
        self.problems = []
        # get path of this file
        import os
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        df = pd.read_csv(f'{path}/raw_files/synthetic_data_biology.csv')
        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break
                correct_answer = str(row['Correct Answer']).replace("\n", "")
                candidates = [correct_answer, str(row['Incorrect Answer 1']).replace("\n", ""),
                              str(row['Incorrect Answer 2']).replace("\n", ""),
                              str(row['Incorrect Answer 3']).replace("\n", "")]
                random.seed(906)
                random.shuffle(candidates)
                label = candidates.index(correct_answer)
                problem = Problem(
                    id=Dataset.generate_hash(row['Question']),
                    question=row['Question'],
                    context=None,
                    label=label,
                    candidates=candidates,
                    explanation=row['Explanation'],
                    problem_index = row['idx'],
                    reference_to=row['reference_to'],
                    reference_type=row['reference_type']
                )
                self.problems.append(problem)
            except Exception as e:
                # print track
                continue