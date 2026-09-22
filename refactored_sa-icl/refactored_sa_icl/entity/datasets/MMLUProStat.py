import ast
import csv
import re
from pathlib import Path
from typing import List

import pandas as pd

from refactored_sa_icl.config.data_paths import MMLU_PRO_STAT_RAW
from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem
def safe_read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        return pd.DataFrame(data)

def _parse_options(raw_text: str) -> list:
    """
    Parse options stored as a string like:
      "['a' 'b' 'c']"  (space-separated, no commas)
    into a Python list.

    Falls back to extracting quoted strings if literal_eval fails.
    """
    if raw_text is None:
        return []

    text = str(raw_text).strip()
    # Convert "['a' 'b']" -> "['a', 'b']" (also supports newlines).
    fixed = re.sub(r"'\s+'", "', '", text)
    fixed = re.sub(r'"\s+"', '", "', fixed)

    try:
        parsed = ast.literal_eval(fixed)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except Exception:
        pass

    # Fallback: extract quoted strings robustly (handles apostrophes inside words).
    matches = re.findall(r"(['\"])(.*?)\1", text, flags=re.DOTALL)
    if matches:
        return [m[1] for m in matches]

    raise ValueError(f"Could not parse options: {raw_text!r}")


class MMLUProStat(Dataset):
    dataset_name: str
    problems: List[Problem]
    size: int

    def __init__(self, size: int = 1_000_000_000):
        self.size = size
        self.load_problems()

    def load_problems(self) -> None:
        self.problems = []

        project_root = Path(__file__).resolve().parents[4]
        csv_path = project_root / MMLU_PRO_STAT_RAW
        df = safe_read_csv(csv_path)

        for i, row in df.iterrows():
            if i >= self.size:
                break

            try:
                candidates = _parse_options(row["options"])
                label = int(row["answer_index"])

                problem = Problem(
                    id=Dataset.generate_hash(row["question"]),
                    question=row["question"],
                    context=None,
                    label=label,
                    candidates=candidates,
                    explanation=None,
                    reference_to=Dataset.generate_hash(row["question"]),
                    reference_type="self",
                )
                self.problems.append(problem)
            except Exception as e:
                print(e)
                break


if __name__ == "__main__":
    dataset = MMLUProStat(size=100)
    for problem in dataset.problems:
        print(problem.get_ground_truth())
    print(len(dataset.problems))
