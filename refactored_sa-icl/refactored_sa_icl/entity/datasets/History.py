from typing import List

import pandas as pd

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class History(Dataset):
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
        df = pd.read_csv(f'{path}/raw_files/history.csv')
        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break
                
                candidates = [str(row['Correct Answer']).replace("\n", "")]
                incorrect_idx = 1
                while f'Incorrect Answer {incorrect_idx}' in row:
                    candidates.append(str(row[f'Incorrect Answer {incorrect_idx}']).replace("\n", ""))
                    incorrect_idx += 1

                label = candidates.index(row['Correct Answer'].replace("\n", ""))
                problem = Problem(
                    id=Dataset.generate_hash(row['Question']),
                    question=row['Question'],
                    context=None,
                    label=label,
                    candidates=candidates,
                    explanation=row['Explanation'],
                    reference_type='history'
                )
                self.problems.append(problem)
            except Exception as e:
                # print track
                print(row)
                print(e)
                continue