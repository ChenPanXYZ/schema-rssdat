import json

from pydantic import BaseModel

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.problems.Problem import Problem

from pydantic import BaseModel, Field
from enum import Enum
from typing import Type, Optional


# Function to dynamically create an Enum
def create_dynamic_enum(name: str, options: list[str]) -> Type[Enum]:
    return Enum(name, {option: option for option in options})

# Example Response model
class Response(BaseModel):
    # reasoning: str
    final_answer: Enum


from typing import Optional, List, Dict, Any


def BaselineSolverUseCase(
        problem: Problem,
        schema: str,
        solver_model,
        knowledges: List[Problem],
        formatted_knowledges: List[Dict[str, Any]],
        settings: Optional[ExperimentSettings] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        *args,
) -> tuple[object, object] | tuple[None, None]:

    # 1. Create Dynamic Response Model
    AnswerEnum = create_dynamic_enum("AnswerEnum", [f"{string}" for string in problem.candidates])

    class DynamicResponse(BaseModel):
        # reasoning: str
        final_answer: AnswerEnum

    # 2. Construct Prompt Content
    prompt_text = "Select the most appropriate answer."

    # We use the multimodal prompt from the problem instance.
    # formatting: [ImageBlock?, TextBlock(Question + Candidates)]
    problem_content = problem.to_prompt(including_answer=False)

    # 3. Build Messages
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": problem_content},
    ]

    # 4. Interact with Model
    response = solver_model.interact(
        messages,
        json_format=DynamicResponse,
        temperature=temperature,
        top_p=top_p
    )

    if response is None:
        return None, None

    return response, response["final_answer"]
