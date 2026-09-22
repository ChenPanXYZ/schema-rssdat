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


def OneShotSchemaSolver(problem,
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
        # reasoning: str
        final_answer: AnswerEnum

    num_shots = settings.solver.num_shots

    # 1. System Prompt
    messages = [{"role": "system", "content": _get_schema_solver_prompt(settings)}]

    # 2. Few-shot Examples (Interleaved)
    shots = knowledges if num_shots is None else knowledges[:num_shots]

    for k in shots:
        # We need a schema for the example. If missing, we skip this example to maintain quality.
        assert k.mental_representation, "Example knowledge must have a mental representation/schema."

        # User: Example Question
        messages.append({"role": "user", "content": k.to_prompt(including_answer=False)})

        # Assistant: Example Schema
        messages.append({"role": "assistant", "content": k.mental_representation})

        # User: Prompt for answer
        messages.append({"role": "user", "content": "Select the most appropriate answer."})

        # Assistant: Example Answer
        # Provide explanation if available to model Chain-of-Thought
        ans_content = f"Answer: {k.get_ground_truth()}"
        # if k.explanation:
        #     ans_content = f"Reasoning: {k.explanation}\n{ans_content}"
        messages.append({"role": "assistant", "content": ans_content})

    # 3. Target Problem
    messages.append({"role": "user", "content": problem.to_prompt(including_answer=False)})

    # Assistant: Target Schema (Pre-calculated and passed in)
    messages.append({"role": "assistant", "content": schema})

    # User: Prompt for answer
    messages.append({"role": "user", "content": "Select the most appropriate answer. Answer with candidate string only, without index."})

    prompt = messages

    response = solver_model.interact(messages, json_format=DynamicResponse, temperature=temperature, top_p=top_p)
    if response is None:
        return None, None
    return response, response["final_answer"]
