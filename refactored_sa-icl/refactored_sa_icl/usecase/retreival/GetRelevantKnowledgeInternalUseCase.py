import numpy as np
import hashlib
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field

from refactored_sa_icl.entity.models.Gemini import Gemini
from refactored_sa_icl.entity.problems.Problem import Problem


# 1. Update the Pydantic model to enforce the 4-option MCQ structure
class DynamicMCQResponse(BaseModel):
    chain_of_thought: str = Field(
        ..., description="Internal reasoning to design a balanced MCQ based on the original problem."
    )
    question_text: str = Field(
        ..., description="The stem of the new multiple-choice question."
    )
    correct_answer: str = Field(
        ..., description="The correct option."
    )
    incorrect_answer_1: str = Field(
        ..., description="A plausible but incorrect distractor."
    )
    incorrect_answer_2: str = Field(
        ..., description="A plausible but incorrect distractor."
    )
    incorrect_answer_3: str = Field(
        ..., description="A plausible but incorrect distractor."
    )
    explanation: str = Field(
        ..., description="Detailed explanation of why the correct answer is right and others are wrong."
    )


def generate_hash(text: str) -> str:
    """Simple hash helper to match your snippet's requirement."""
    return hashlib.md5(text.encode()).hexdigest()


def GetRelevantKnowledgeInternalUseCase(
        problem: Problem,
        solver_model
) -> Tuple[List[Problem], None]:
    """
    Retrieves a Self-Generated MCQ Example.
    Returns a Problem object populated with metadata suitable for your df_field construction.
    """
    temperature: float = 0.7
    top_p: float = 0.95
    ref_type = "internal"

    # 2. Refined Prompt for MCQ Generation
    system_prompt = (
        "You are an expert examiner. Your task is to analyze a given problem and "
        "generate a high-quality Multiple Choice Question (MCQ) that tests the same "
        "underlying logic or concept."
    )

    _ = "1. Create a NEW MCQ with 4 options (1 correct, 3 distractors) based on the logic of the problem above.\n""2. Ensure the distractors are plausible common mistakes.\n""3. Provide a clear explanation."

    user_content = []
    user_content.append({"type": "text", "text": "Original Problem:\n"})
    user_content.append(problem.to_prompt(including_answer=False))
    user_content.append({"type": "text", "text": _})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

     #let's use gemini
    gemini = Gemini(4096)

    # 3. API Call
    response_obj: DynamicMCQResponse = gemini.interact(
        messages,
        json_format=DynamicMCQResponse,
        temperature=temperature,
        top_p=top_p
    )

    # 4. Construct the Problem Object
    # We pack the specific MCQ fields into a dictionary (metadata) or attributes
    # so they can be easily extracted for your df_field later.

    # Pre-calculating the dictionary structure you requested to ensure compatibility
    # This stores the exact dict you wanted into a 'metadata' attribute (or similar)
    id = generate_hash(response_obj['question_text'])
    generated_problem = Problem(
                    id=id,
                    question=response_obj['question_text'],
                    context=None,
                    label=3,
                    candidates=[response_obj['incorrect_answer_1'],response_obj['incorrect_answer_2'],response_obj['incorrect_answer_3'], response_obj['correct_answer']],
                    explanation=response_obj['explanation'],
                    reference_to=problem.id,
                    reference_type="internal"
                )
    return [generated_problem], None