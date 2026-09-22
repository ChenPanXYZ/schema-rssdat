import hashlib

import pandas as pd

from refactored_sa_icl.entity.datasets.MedXpertQA_Skeletal import MedXpertQA_Skeletal
from refactored_sa_icl.entity.datasets.CommonSenseQA import CommonSenseQA
from refactored_sa_icl.entity.datasets.GPQA import GPQA
from refactored_sa_icl.entity.datasets.GPQABiology import GPQABiology
from refactored_sa_icl.entity.datasets.GPQAPhysics import GPQAPhysics
# from refactored_sa_icl.entity.datasets.HLEChemistry import HLEChemistry`
# from refactored_sa_icl.entity.datasets.HLEMath import HLEMath
from refactored_sa_icl.entity.models.GPT_Parse import GPT_Parse
from refactored_sa_icl.entity.models.GPT4o import GPT4o
from refactored_sa_icl.entity.datasets.Math import Math
from logging import getLogger
import logging

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

import os
from pydantic import BaseModel


class QuestionOutput(BaseModel):
    question: str
    answer: str
    explanation: str


QUESTION_FORMAT = """
You should output a json object with the following format:
{
    "question": "Question text...",
    "answer": "Answer text. Just include the final answer only.",
    "explanation": "Provide a short explanation of the answer, with the reasoning process, and the knowledge required to answer the question."
}
"""

# call gpt-4o-mini to generate a synthetic data for each problem.
# system_prompt = """
# You are a subject test expert that is designing a new graduate level test multiple choice question provided with a reference question and its answer.
# Each new question should be designed in a similar difficulty to the reference question, and should follow certain additional criterias.
# """


prompt = {
    "paraphrase": """
Consider this question: {question}, along with its answer({answer}) and the explanation of solving it({explanation}).

Please paraphrase the given question to test the student's ability to answer the same question with different wording, to 
evaluate whether they can answer an almost identical question they have seen before.

You should follow the following criteria: 
- It is only worded differently from the original question. 
- Provide a short explanation similar to the given explanation on how to solve the new question

- Output Format:
    {question_format}
""",

    "new_question": """
Consider this question: {question}, along with its answer({answer}) and the explanation of solving it({explanation}).

Please give me a slightly different question from this example that test the student's ability to transform their knowledge.

You should follow the following criteria:
- The new question only requires the knowledge provided in the explanation to be used to answer it.
- New question should still differ with a lot of distinctiveness to test student's use of the same knowledge.
- Provide a short explanation on how to solve the new question
- Difficulty: 
    The new question should be the similar difficulty to the previous question.
    If a student has the knowledge to answer the previous question, they should have enough knowledge to answer the new question.
- Distinctiveness: 
    The new question should be distinctive enough to the previous question, that the student cannot use the same answer.
    New question should be unique in its context, but still related to the previous question.
- Output Format:
    {question_format}

""",

    "new_question_exam": """
Consider this question: {question}, along with its answer({answer}) and the explanation of solving it({explanation}).

Please generate a new question that is distinct from the previous question.

You should follow the following criteria:
- New question requires more knowledge than the provided explanation to be used to answer it.
- New question should differ from the given question with a lot of distinctiveness.
- Provide a short explanation on how to solve the new question, and the additional knowledge required to answer the new question.
- Difficulty:
    The new question should be the similar difficulty to the previous question.
    If a student has the knowledge to answer the previous question, they should have partial knowledge to answer the new question.
    However, the new question should require additional knowledge than the given question's scope to be answered.
- Distinctiveness:
    The new question should be distinctive enough to the previous question, that the student require additional knowledge to solve the problem. 
    New question should be unique in its context, and is related to the previous question in a minimal level.
- Output Format:
    {question_format}
"""
}

def generate_hash(input_string):
    """
    Generate a 36-character alphanumeric hash from the input string.

    Args:
        input_string (str): The string to hash.

    Returns:
        str: A 36-character hash.
    """
    # Generate a SHA-256 hash of the input string
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()

    # Truncate or extend the hash to 36 characters
    hash_36 = sha256_hash[:36]  # Take the first 36 characters

    return hash_36


def main(args):
    # check if synthetic_wip.csv exists.
    if os.path.exists(args.output_file):
        df = pd.read_csv(args.output_file)
    else:
        df = pd.DataFrame()

    dataset = Math(size=100000000)
    model = GPT_Parse()
    for idx, problem in enumerate(dataset.problems):

        for prompt_type in prompt.keys():
            # check if problem.id and prompt_type already exists in the dataframe.
            if len(df) != 0 and df[(df['reference_to'] == problem.id) & (df['reference_type'] == prompt_type)].shape[
                0] > 0:
                print("done")
                continue
            synthetic_data = model.interact(
                model="gpt-4o",
                messages=prompt[prompt_type].format(
                    question=problem.question,
                    answer=problem.label,
                    explanation=problem.explanation,
                    question_format=QUESTION_FORMAT
                ),
                text_format=QuestionOutput,
                temperature=0
            )

            idx = idx
            type = prompt_type
            data = synthetic_data

            # add the data to the csv file
            df_field = {
                'id': generate_hash(data['question']),
                'Question': [data['question']],
                'Correct Answer': [data['answer']],
                'Explanation': [data['explanation']],
                'idx': [idx],
                'reference_to': problem.id,
                'reference_type': type
            }
            new_row = pd.DataFrame(df_field)
            df = pd.concat([df, new_row], ignore_index=True)
            # save the synthetic data to a file.
            try:
                df.to_csv(args.output_file, index=False, escapechar='\\')
            except:
                logger.error("Error saving synthetic data to file")
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="synthetic_wip.csv")
    args = parser.parse_args()
    main(args)
