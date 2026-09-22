import os
import ast
import traceback
import pandas as pd
from typing import List

# Assuming these imports exist in your project structure
from refactored_sa_icl.entity.datasets.Dataset import Dataset
from refactored_sa_icl.entity.problems.Problem import Problem


class Safety(Dataset):
    dataset_name: str = "Safety"
    problems: List[Problem]
    size: int

    def __init__(self, size: int = 1000000000):
        self.size = size
        self.load_problems()

    def load_problems(self):
        self.problems = []
        # Get path of this file to find the raw_files folder
        path = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(path, 'raw_files/raw_files/safety_questions.csv')

        # Check if file exists to avoid confusion
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        for i, row in df.iterrows():
            try:
                if i >= self.size:
                    break

                # 1. Handle 'choices' column
                # It is likely stored as a string representation of a list: "['A', 'B']"
                choices_raw = row['choices']
                if isinstance(choices_raw, str):
                    # safely evaluate the string literal to a list
                    candidates = ast.literal_eval(choices_raw)
                elif isinstance(choices_raw, list):
                    candidates = choices_raw
                else:
                    candidates = []

                # 2. Handle 'image' column
                # The parquet file likely saved the image dict as a string
                image_raw = row['image']
                if isinstance(image_raw, str):
                    try:
                        # Try parsing as a dictionary string first
                        image_data = ast.literal_eval(image_raw)
                    except (ValueError, SyntaxError):
                        # Fallback: if it's just a path string or raw bytes string that failed eval
                        image_data = image_raw
                else:
                    image_data = image_raw

                # 3. Create Problem Instance
                # We map the CSV columns to the Problem fields
                problem = Problem(
                    id=str(row['id']),
                    question=row['question'],
                    context=image_data,  # Contains the bytes or path
                    label=int(row['label']),
                    candidates=candidates,
                    # Fallback to empty string if 'explanation' is missing or NaN
                    explanation=str(row.get('explanation', "")) if pd.notna(
                        row.get('explanation')) else "",
                    reference_to=str(row['id']),
                    reference_type="self"
                )
                self.problems.append(problem)

            except Exception as e:
                print(f"Error processing row {i}: {e}")
                traceback.print_exc()
                continue

        print(f"Successfully loaded {len(self.problems)} problems.")


if __name__ == "__main__":
    # Test the loader
    dataset = Safety(size=10000)
    # check if any rpoblems has label greater than length of candidates
    for problem in dataset.problems:
        if problem.label >= len(problem.candidates):
            print(f"Problem ID {problem.id} has invalid label {problem.label} for candidates {problem.candidates}")
            # get index
            index = dataset.problems.index(problem)
            print(f"Index: {index}")
