import base64
from typing import List

from pylatexenc.latex2text import LatexNodes2Text

from refactored_sa_icl.usecase.solver.utils import _load_reasoning_for_id

import re


def extract_boxed_answer(text):
    """Extracts the content inside \boxed{}."""
    if "\\boxed{" in text:
        # Finds the content inside the last boxed{} block
        parts = text.split("\\boxed{")
        if len(parts) > 1:
            ans = parts[-1]
            line = ""
            count = 1
            for char in ans:
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1
                if count == 0:
                    break
                line += char
            return line
    return None


def is_equiv(model_output, ground_truth):
    """Basic normalization for equivalence checking."""
    # 1. Extract boxed answers from both if necessary
    model_ans = extract_boxed_answer(model_output) or model_output

    # 2. Simple Normalization (Lowercasing, stripping whitespace)
    model_ans = str(model_ans).strip().lower()
    ground_truth = str(ground_truth).strip().lower()

    # 3. Numeric comparison (optional but recommended for MATH)
    try:
        if float(model_ans) == float(ground_truth):
            return True
    except:
        pass

    return model_ans == ground_truth

class Problem:
    """
    The Problem Object for this framework.
    """
    question: str
    context: str
    label: int
    candidates: list[str]
    explanation: str

    '''
    question: str. The question boday.
    context: str. The context of the question.
    label: int. The index of the correct answer in the candidates list. (optional only if the problem is close-ended)
    candidates: list[str]. The list of candidate solutions. (optional only if the problem is close-ended)
    '''
    def __init__(self, id, question, context = None, label = None, candidates = None, explanation = None, problem_index =
    None, reference_to = None, reference_type = None):
        self.id = id
        self.context = context
        self.question = LatexNodes2Text().latex_to_text(question)
        self.label = label # this is the index of the correct answer in the candidates list
        if candidates is not None:
            for i in range(len(candidates)):
                if type(candidates[i]) != str:
                    candidates[i] = str(candidates[i])
                else:
                    candidates[i] = LatexNodes2Text().latex_to_text(candidates[i])
        self.candidates = candidates
        self.explanation = explanation
        self.problem_index = problem_index
        self.mental_representation = None
        self.reference_to = reference_to
        self.reference_type = reference_type
        self.embedding = None

    def add_embedding(self, embedding):
        self.embedding = embedding


    def __str__(self):
        raise NotImplementedError
        # result = self.id + "\n"
        result = ""
        if self.context:
            result += f"Context: {self.context}\n"

        result += f"Question: {self.question}\n"

        if self.candidates:
            candidates = "\n" + "\n".join(
            f"{i+1}. {string}" for i, string in enumerate(self.candidates)
        )
            result += f"Candidates: {candidates}"

        return result

    import base64

    def to_prompt(self, including_answer, reasoning=False) -> List:
        content = []

        # 1. Handle the Question and Candidates
        text_query = f"Question: {self.question}"
        if self.candidates:
            candidates_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.candidates))
            text_query += f"\nCandidates:\n{candidates_str}"
        text_query += "\n"
        content.append({"type": "text", "text": text_query})


        # 1. Handle the Image Context
        if self.context:
            # Check if context is the dictionary with bytes you showed earlier
            if isinstance(self.context, dict) and 'bytes' in self.context:
                image_bytes = self.context['bytes']
                base64_image = base64.b64encode(image_bytes).decode('utf-8')

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
            else:
                # If context is just text (like a URL or description), add as text
                content.append({"type": "text", "text": f"Image: {self.context}"})

        if including_answer:
            content.append({"type": "text", "text": f"Answer to this question is: {self.get_ground_truth()}"})

        if reasoning:
            # Get reasoning from reasoning map
            # example_reasoning = _load_reasoning_for_id(self.id)
            # if (example_reasoning is None or example_reasoning.strip() == "") and getattr(self, "explanation", None):
            #     example_reasoning = str(self.explanation)
            # if example_reasoning is None:
            #     example_reasoning = "Reasoning not available."
            #     raise ValueError(f"Example reasoning requested but not found for question ID {self.id}")
            # else:
            #     content.append({"type": "text", "text": f"\n\nExample Reasoning:\n{example_reasoning}\n"})
            assert self.mental_representation is not None, "Mental representation must be generated before including reasoning."
            # TODO: New logic, always use the CoT mental representation!
            # filtered_lines = [line for line in self.mental_representation.splitlines() if
            #                   not line.strip().startswith('#')]
            #
            # # Rejoin the remaining lines
            # example_reasoning = "\n".join(filtered_lines).strip()
            # content.append({"type": "text", "text": f"\nExample Reasoning:\n{example_reasoning}\n"})
            _ = self.mental_representation.replace('#### Schema:\n##### ', "")
            content.append({"type": "text", "text": f"\nExample Reasoning:\n{_}\n"})


        return content

    def get_ground_truth(self):
        if self.candidates is None:
            return self.label
        else:
            return self.candidates[self.label]

    def get_output_format(self):
        if self.candidates is None:
            return {
                "type": "string",
                "description": "The final answer"
            }
        else:
            return {
                "type": "string",
                "description": "The chosen candidate solution",
                "enum": [f"{string}" for string in self.candidates]
            }

    def evaluate_open_ended(self, answer):
        raise NotImplementedError

    def evaluate_closed_ended(self, answer) -> bool:
        ground_truth = self.get_ground_truth()
        return answer == ground_truth

    def evaluate_non_mcq(self, answer) -> bool:
        return is_equiv(answer, self.label)

    def evalute(self, answer):
        if self.label is None:
            return self.evaluate_open_ended(answer)
        elif self.candidates is None:
            return self.evaluate_non_mcq(answer)
        else:
            return self.evaluate_closed_ended(answer)

    def if_has_context(self):
        return self.context is not None and self.context != ""


    def get_image_payload(self):
        if not self.context or 'bytes' not in self.context:
            return None


        b64_str = base64.b64encode(self.context['bytes']).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_str}"}
        }

    # Inside your Problem class:
    def to_dict(self):
        data = vars(self).copy()

        # Handle the 'context' if it contains image bytes
        if isinstance(self.context, dict) and 'bytes' in self.context:
            # Convert bytes to base64 string so it can be saved in JSON/DB
            image_bytes = self.context['bytes']
            data['context'] = {
                'bytes': base64.b64encode(image_bytes).decode('utf-8'),
                'format': self.context.get('format', 'png')
            }

        # Remove any non-serializable objects (like internal model references) if they exist
        # e.g., if you had self.embedder, you'd delete it here:
        # data.pop('embedder', None)

        return data
