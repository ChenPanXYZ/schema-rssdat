from typing import List
import random
import pandas as pd

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem

from huggingface_hub import login
import os
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

class HLEChemistry(Dataset):
    dataset_name: str
    problems: List[Problem]
    size: int
    def __init__(self, size:int= 1000000000):
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
        # df = load_dataset("cais/hle", cache_dir=root.root + '/.cache/huggingface')['test'].to_pandas()
        # df.to_csv("HLE_FULL.csv", index=False)
        import os
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        df = pd.read_csv(f'{path}/raw_files/HLE_Chemistry.csv')
        for i, row in df.iterrows():
            question = row['Question']
            # Answer Choices is a list string representation
            correct_answer = str(row['Correct Answer']).replace("\n", "")
            candidates = [correct_answer, str(row['Incorrect Answer 1']).replace("\n", ""),
                            str(row['Incorrect Answer 2']).replace("\n", ""),
                            str(row['Incorrect Answer 3']).replace("\n", "")]
            random.seed(906)
            random.shuffle(candidates)
            label = candidates.index(correct_answer)
            problem = Problem(
                id=Dataset.generate_hash(question),
                question=question,
                context=None,
                label=candidates.index(row['Correct Answer']),
                candidates=candidates,
                explanation=row['Explanation'],
                reference_to=Dataset.generate_hash(question),
                reference_type="self"
            )
            self.problems.append(problem)


if __name__ == "__main__":
    # test the dataset
    dataset = HLEChemistry()
    print(len(dataset.problems))
    for problem in dataset.problems:
        print(problem.explanation)
        break
