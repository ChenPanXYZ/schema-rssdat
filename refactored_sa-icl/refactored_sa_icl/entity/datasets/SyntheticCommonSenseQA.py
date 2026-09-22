import json
from typing import List
import os
import csv
import random

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class SyntheticCommonSenseQA(Dataset):
    dataset_name: str
    problems: List[Problem]
    size: int

    def __init__(self, size: int = 1000000000):
        self.size = size
        self.load_problems()

    '''
    Load Problem from dataset.
    Please refer to the Problem class to see which columns are required.
    '''
    def load_problems(self):
        self.problems = []
        # Resolve path to raw_files/synthetic_data_MedXpertQA.csv next to this file
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        data_path = os.path.join(path, "raw_files", "synthetic_commonsenseqa.csv")

        count_loaded = 0
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if count_loaded >= self.size:
                    break
                if row is None:
                    continue

                # Basic fields
                question_text = (row.get("Question") or "").strip()
                explanation = (row.get("Explanation") or "").strip()

                # Candidates: place correct answer first so label = 0
                correct_answer = (row.get("Correct Answer") or "").replace("\n", " ").strip()
                incorrect_1 = (row.get("Incorrect Answer 1") or "").replace("\n", " ").strip()
                incorrect_2 = (row.get("Incorrect Answer 2") or "").replace("\n", " ").strip()
                incorrect_3 = (row.get("Incorrect Answer 3") or "").replace("\n", " ").strip()
                incorrect_4 = (row.get("Incorrect Answer 4") or "").replace("\n", " ").strip()

                # Build candidates list, preserving order [Correct, Incorrect1, Incorrect2, Incorrect3, Incorrect4]
                candidates = [ans for ans in [correct_answer, incorrect_1, incorrect_2, incorrect_3, incorrect_4] if ans != ""]
                random.seed(906)
                random.shuffle(candidates)
                label_index = candidates.index(correct_answer)

                # Optional metadata
                id_field = (row.get("id") or "").strip()
                problem_id = id_field if id_field else Dataset.generate_hash(question_text)

                # problem_index (idx) if parsable
                idx_raw = row.get("idx")
                try:
                    problem_index = int(idx_raw) if idx_raw not in (None, "") else None
                except ValueError:
                    problem_index = None

                reference_to = (row.get("reference_to") or "").strip() or problem_id
                reference_type = (row.get("reference_type") or "").strip() or "self"

                # Skip if no question or no candidates
                if question_text == "" or not candidates:
                    continue

                problem = Problem(
                    id=problem_id,
                    question=question_text,
                    context=None,
                    label=label_index,
                    candidates=candidates,
                    explanation=explanation if explanation != "" else None,
                    problem_index=problem_index,
                    reference_to=reference_to,
                    reference_type=reference_type
                )
                self.problems.append(problem)
                count_loaded += 1


if __name__ == "__main__":
    dataset = SyntheticCommonSenseQA(size=100000)
    for problem in dataset.problems:
        print(problem)
        print(problem.get_ground_truth())
