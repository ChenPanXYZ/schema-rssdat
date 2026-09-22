import random
import os
import ast
from typing import List
import pandas as pd
import traceback

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class Vision(Dataset):
    dataset_name: str = "Vision"
    problems: List[Problem]
    size: int

    def __init__(self, size: int = 1000000000):
        self.size = size
        self.load_problems()

    def load_problems(self):
        self.problems = []
        # Get path of this file to find the raw_files folder
        path = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(path, 'raw_files/vision_questions.csv')

        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break

                # Handle 'choices' column: Convert string representation of list to actual list
                choices_raw = row['choices']
                if isinstance(choices_raw, str):
                    candidates = ast.literal_eval(choices_raw)
                else:
                    candidates = choices_raw

                # The 'image' column contains a dictionary string with 'bytes'.
                # We extract the byte data to use as the context.
                image_data = eval(row['image'])
                id = row['id']
                problem = Problem(
                    id=id,
                    question=row['question'],
                    context=image_data,  # Passing the image dictionary/bytes string here
                    label=int(row['label']),
                    candidates=candidates,
                    explanation=row.get('description', ""),  # Using description as explanation
                    reference_to=str(row['id']),
                    reference_type="self"
                )
                self.problems.append(problem)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                # p[rint trace
                traceback.print_exc()
                continue


if __name__ == "__main__":
    dataset = Vision(size=10)
    print(dataset.problems)
    for problem in dataset.problems:
        print(problem)
        print("----")
        break
