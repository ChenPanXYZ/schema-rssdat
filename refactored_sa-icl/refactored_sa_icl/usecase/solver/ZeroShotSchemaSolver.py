import json

import json
from enum import Enum
from typing import Type, Optional

from pydantic import BaseModel

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.usecase.solver.ComprehensiveSchemaSolver import _get_schema_solver_prompt


def create_dynamic_enum(name: str, options: list[str]) -> Type[Enum]:
    return Enum(name, {option: option for option in options})


def ZeroShotSchemaSolver(problem,
    schema,
    solver_model,
    knowledges,
    formatted_knowledges,
    settings: Optional[ExperimentSettings] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    *args,):
    global prompt
    AnswerEnum = create_dynamic_enum("AnswerEnum", [f"{string}" for string in problem.candidates])

    class DynamicResponse(BaseModel):
        reasoning: str
        final_answer: AnswerEnum

    num_shots = settings.solver.num_shots

    # 1. System Prompt
    messages = [{"role": "system", "content": _get_schema_solver_prompt(settings)}]

    # 3. Target Problem
    messages.append({"role": "user", "content": problem.to_prompt(including_answer=False)})

    # Assistant: Target Schema (Pre-calculated and passed in)
    messages.append({"role": "assistant", "content": schema})

    # User: Prompt for answer
    messages.append({"role": "user", "content": "Select the most appropriate answer."})

    prompt = messages

    response = solver_model.interact(messages, json_format=DynamicResponse, temperature=temperature, top_p=top_p)
    if response is None:
        return None, None
    return response, response["final_answer"]
