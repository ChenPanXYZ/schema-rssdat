import json
from typing import List
import os

from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class MedXpertQA_Skeletal(Dataset):
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
        # Resolve path to raw_files/MedXpertQA.jsonl next to this file
        path = os.path.abspath(__file__)
        path = os.path.dirname(path)
        data_path = os.path.join(path, "raw_files", "MedXpertQA_Skeletal.jsonl")

        count_loaded = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if count_loaded >= self.size:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                question_text = record.get("question", "")
                options_dict = record.get("options", {}) or {}
                label_letter = record.get("label")
                medical_task = record.get("medical_task", "")
                body_system = record.get("body_system", "")
                question_type = record.get("question_type", "")

                # Preserve candidate order as in the JSONL: A, B, C, ...
                # Build candidates by iterating option letters in sorted alphabetical order
                # which mirrors the canonical display order in the source.
                option_letters = sorted(options_dict.keys())
                candidates = [str(options_dict[letter]).replace("\n", "").replace('"', "'") for letter in option_letters]

                # Convert letter label to candidate index
                if label_letter is None:
                    label_index = None
                else:
                    try:
                        label_index = option_letters.index(label_letter)
                    except ValueError:
                        # If an unexpected label is found, skip this record
                        continue

                # Explanation placeholder per requirement
                explanation = f"[{medical_task}] - [{body_system}] - [{question_type}]"

                problem = Problem(
                    id=Dataset.generate_hash(question_text),
                    question=question_text,
                    context=None,
                    label=label_index,
                    candidates=candidates,
                    explanation=explanation,
                    reference_to=Dataset.generate_hash(question_text),
                    reference_type="self"
                )
                self.problems.append(problem)
                count_loaded += 1


if __name__ == "__main__":
    dataset = MedXpertQA_Skeletal(size=500000)
    print(len(dataset.problems))

