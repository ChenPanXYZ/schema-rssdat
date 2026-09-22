import ast
import random
from typing import List

from pathlib import Path
import pandas as pd
import re

from refactored_sa_icl.config.data_paths import MMLU_BUSINESS_RAW
from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem

import re


# TODO: see if we need to use it for mmlu math as well.
def parse_text_to_array(raw_text):
    """
    Converts a string representation of a list (with mixed quotes)
    into a standard Python list of strings.
    """
    # Pattern: Finds content between matching single (') or double (") quotes
    # Handles "Company's" correctly by matching the outer wrapper
    pattern = r"(['\"])(.*?)\1"

    # Extract the content from the capture group (index 1 of the match tuple)
    matches = re.findall(pattern, raw_text)
    return [m[1] for m in matches]

class MMLUBusiness(Dataset):
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
        csv_path = project_root / MMLU_BUSINESS_RAW
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

                candidates = parse_text_to_array(row['options'])



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
    dataset = MMLUBusiness(size=100000)
    for problem in dataset.problems:
        try:
            print(problem)
            print(problem.get_ground_truth())

        except Exception as e:
            print(problem.candidates)
            print(len(problem.candidates))
            print(problem.label)
            print(f"Error in problem ID {problem.id}: {e}")
    print(len(dataset.problems))
