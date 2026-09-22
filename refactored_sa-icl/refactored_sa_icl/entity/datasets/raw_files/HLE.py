# from typing import List
# from datasets import load_dataset
#
# import root
# import string
#
# from refactored_sa_icl.entity.datasets.Dataset import Dataset
# from refactored_sa_icl.entity.problems.Problem import Problem
#
# from huggingface_hub import login
#
#
#
# class HLE(Dataset):
#     dataset_name: str
#     problems: List[Problem]
#     size: int
#
#     def __init__(self, size: int = 1000000000):
#         self.size = size
#         self.load_problems()
#
#     '''
#     Load Problem from dataset.
#     Please refer to the Problem class to see which columns are required.
#     '''
#
#     def load_problems(self):
#         results = []
#         # load csv gpqa_main.csv
#         self.problems = []
#         # get path of this file
#         df = load_dataset("cais/hle", cache_dir=root.root + '/.cache/huggingface')['test'].to_pandas()
#
#         df.to_csv("HLE_FULL.csv", index=False)
#
#         count = 0
#         for i, row in df.iterrows():
#             # try:
#             # Error Checking
#             if row['image'].strip() != "":
#                 continue
#             elif row['rationale_image'] is not None:
#                 continue
#             elif row['answer_type'] != 'multipleChoice':
#                 continue
#             # Check for size
#             if count >= self.size:
#                 break
#
#             # Valid Data, add one to our problemset
#             count += 1
#             # Parse the question
#             question = row['question']
#             answer_idx = question.find('Answer Choices:')
#             question = question[:answer_idx].strip()
#
#             # Parse the answer
#             answers = row['question'].split('Answer Choices:')[1].split('\n')
#             candidates = [answer[3:] for answer in answers if answer.strip() != '']
#             correct_answer = row['answer'].lower()
#             label = string.ascii_lowercase.index(correct_answer)
#
#             problem = Problem(
#                 id=Dataset.generate_hash(question),
#                 question=question,
#                 context=None,
#                 label=label,
#                 candidates=candidates,
#                 explanation=row['rationale'],
#                 reference_to=Dataset.generate_hash(question),
#                 reference_type="self"
#             )
#             # add to a csv dataframe
#             results.append({
#                 'id': problem.id,
#                 'question': problem.question,
#                 'candidates': problem.candidates,
#                 'label': problem.label,
#                 'explanation': problem.explanation,
#                 'reference_to': problem.reference_to,
#                 'reference_type': problem.reference_type
#             })
#             self.problems.append(problem)
#
#         # Save to a csv file
#         import pandas as pd
#         df = pd.DataFrame(results)
#         #df.to_csv(f'{root.root}/src/entity/datasets/raw_files/HLE.csv', index=False)
#
#
# if __name__ == "__main__":
#     hle = HLE()
#     print(len(hle.problems))
