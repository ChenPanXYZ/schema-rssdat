from typing import List
import random
from pathlib import Path
import pandas as pd

from refactored_sa_icl.config.data_paths import MMLU_PRO_FINANCE_SYNTHETIC_RAW
from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class MMLUProFinanceSynthetic(Dataset):
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

        project_root = Path(__file__).resolve().parents[4]
        csv_path = project_root / MMLU_PRO_FINANCE_SYNTHETIC_RAW
        df = pd.read_csv(csv_path)
        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break

                correct_answer = str(row['Correct Answer']).replace("\n", "")
                candidates = [correct_answer, str(row['Incorrect Answer 1']).replace("\n", ""),
                              str(row['Incorrect Answer 2']).replace("\n", ""),
                              str(row['Incorrect Answer 3']).replace("\n", "")]
                # if config.get("shuffle", False):
                #     random.seed(906)
                #     random.shuffle(candidates)
                label = candidates.index(correct_answer)

                label = candidates.index(row['Correct Answer'].replace("\n", ""))
                problem = Problem(
                    id=Dataset.generate_hash(row['Question']),
                    question=row['Question'],
                    context=None,
                    label=label,
                    candidates=candidates,
                    explanation=row['Explanation'],
                    problem_index=row['idx'],
                    reference_to=row['reference_to'],
                    reference_type=row['reference_type']
                )
                self.problems.append(problem)
            except Exception as e:
                # print track
                print(row)
                print(e)
                continue

if __name__ == "__main__":
    dataset = MMLUProFinanceSynthetic(size=10)
    for problem in dataset.problems:
        print(problem)