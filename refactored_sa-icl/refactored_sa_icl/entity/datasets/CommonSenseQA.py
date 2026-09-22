import json
from typing import List
import os

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


import ast
import random
from typing import List

from pathlib import Path
import pandas as pd
import re

from refactored_sa_icl.config.data_paths import MMLU_COLLEGE_MATH_RAW
from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class CommonSenseQA(Dataset):
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

        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        data_path = os.path.join(path, "raw_files", "commonsenseqa.csv")
        df = pd.read_csv(data_path)
        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break

                candidates = ast.literal_eval(row['candidates'])
                answer_index = candidates.index(row['answer'])
                assert answer_index is not None
                print(candidates)
                print(answer_index)



                problem = Problem(
                    id=Dataset.generate_hash(row['Question']),
                    question=row['Question'],
                    context=None,
                    label=answer_index,
                    candidates=candidates,
                    explanation=None,
                    reference_to=Dataset.generate_hash(row['Question']),
                    reference_type="self"
                )
                self.problems.append(problem)
            except Exception as e:
                print(e)
                break

if __name__ == "__main__":
    dataset = CommonSenseQA(size=10000)
    for problem in dataset.problems:
        print("hi")

    print(len(dataset.problems))




# import pandas as pd
# from datasets import load_dataset
#
# ds = load_dataset("tau/commonsense_qa")
#
# # Randomly pick 200 examples from the *test* split (example)
# sample = ds["validation"].shuffle(seed=906).select(range(200))
#
# results = []
# for q in sample:
#     labels = q['choices']['label']
#     texts = q['choices']['text']
#     answerKey = q['answerKey']
#     answerIndex = labels.index(answerKey)
#     assert answerIndex is not None
#
#     answer = texts[answerIndex]
#     results.append(
#         {
#             "Question": q['question'],
#             "answer": answer,
#             "candidates": texts
#         }
#     )
#
# pd.DataFrame(data = results).to_csv("raw_files/commonsenseqa.csv", index = False)
