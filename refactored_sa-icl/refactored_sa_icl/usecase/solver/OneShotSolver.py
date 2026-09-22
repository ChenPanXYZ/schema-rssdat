import json

import json
import os
from enum import Enum
from typing import Type, Optional

import pandas as pd
from pydantic import BaseModel

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.usecase.solver.utils import get_role_playing_text
from refactored_sa_icl.usecase.solver.utils import get_one_shot_system_prompt

def create_dynamic_enum(name: str, options: list[str]) -> Type[Enum]:
    return Enum(name, {option: option for option in options})


def OneShotSolver(problem,
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

    if settings is not None:
        num_shots = settings.solver.num_shots
        with_example_reasoning = settings.solver.with_example_reasoning
    else:
        num_shots = 1
        with_example_reasoning = False

    include_answer_in_example = settings.solver.include_answer_in_example
    assert include_answer_in_example is not None, "include_answer_in_example must be specified in settings for OneShotSolver."

    system_prompt = get_one_shot_system_prompt(include_answer_in_example, with_example_reasoning)

    messages = [{"role": "system", "content": system_prompt}]
    # 2. Few-shot Examples (Interleaved)
    shots = knowledges if num_shots is None else knowledges[:num_shots]

    example_strings = []
    # example_strings.append({"type": "text", "text": "First, you are given an example question along with its solution.\n"})
    if include_answer_in_example and with_example_reasoning:
        first = "Below is an example question along with its solution and reasoning"
    elif include_answer_in_example and not with_example_reasoning:
        first = "Below is an example question along with its solution"
    elif not include_answer_in_example and with_example_reasoning:
        first = "Below is an example question along with its reasoning"
    else:
        raise ValueError("Invalid configuration: include_answer_in_example and with_example_reasoning cannot both be False for OneShotSolver.")

    example_strings.append({"type": "text", "text": first + ":\n"})
    for k in shots:
        # We need a schema for the example. If missing, we skip this example to maintain quality.
        # User: Example Question
        example_strings.extend(k.to_prompt(including_answer=(True and include_answer_in_example), reasoning=with_example_reasoning))
        # if k.explanation:
        #     ans_content = f"Reasoning: {k.explanation}\n{ans_content}"

    assert example_strings, "No valid few-shot examples available for OneShotSolver."
    messages.append({"role": "user", "content": example_strings})



    # 3. Target Problem
    new_question_prompts = []
    new_question_prompts.append({"type": "text", "text": "Below is the **new question**:\n"})
    # if with_example_reasoning:
    #     new_question_prompts.append({"type": "text", "text": "Think about how you could use the example along with its solution and reasoning to solve the **new question**:\n"})

    new_question_prompts.extend(problem.to_prompt(including_answer=False))

    new_question_prompts.append({"type": "text", "text": "Solve the new question. Select the most appropriate answer."})
    messages.append({"role": "user", "content": new_question_prompts})

    # User: Prompt for answer
    prompt = messages

    response = solver_model.interact(messages, json_format=DynamicResponse, temperature=temperature, top_p=top_p)
    if response is None:
        return None, None
    return response, response["final_answer"], messages
