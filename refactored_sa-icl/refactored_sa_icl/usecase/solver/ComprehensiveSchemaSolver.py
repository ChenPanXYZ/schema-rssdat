import json
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Type

from pydantic import BaseModel

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.problems.Problem import Problem


def create_dynamic_enum(name: str, options: list[str]) -> Type[Enum]:
    return Enum(name, {option: option for option in options})


def _get_schema_solver_prompt(settings: Optional[ExperimentSettings]) -> str:
    """
    Fetch the schema-solver prompt from ExperimentSettings.extra if present,
    otherwise fall back to a generic instruction.
    """
    if settings is not None and "SCHEMA_SOLVER_PROMPT" in settings.extra:
        return str(settings.extra["SCHEMA_SOLVER_PROMPT"])

    raise ValueError(
        "SCHEMA_SOLVER_PROMPT not found in settings.extra. Please provide a valid prompt."
    )


def ComprehensiveSchemaSolver(
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

    AnswerEnum = create_dynamic_enum(
        "AnswerEnum", [f"{string}" for string in problem.candidates]
    )

    class DynamicResponse(BaseModel):
        # reasoning: str
        final_answer: AnswerEnum

    assert formatted_knowledges, "formatted_knowledges (multimodal list) must be provided."


    # 1. Prepare the Current Problem (User Message 1)
    current_problem_content = problem.to_prompt(including_answer=False)

    current_problem_content.append({"type": "text", "text": "Below is a preliminary schema\n"})

    current_problem_content.append({"type": "text", "text": problem.mental_representation})

    # 3. Construct the Conversation History
    # system_prompt = (
    #     "You are given an example question along with its solution and schema. "
    #     "Then, you will be given a new question with a preliminary schema. "
    #     "Your task is to refine the preliminary schema based on the example provided, "
    #     "and then use the refined schema to answer the new question."
    # )
    system_prompt ="You will be givin an example question along with its schema. Then, you will be given a new question with a preliminary schema. Think about if you could refine the preliminary schema based on the example question. Then, select the most appropriate answer for the new question."


    # Combine: [Intro] + [Past Knowledge List] + [Outro]
    refinement_content = formatted_knowledges + [
        {"type": "text", "text": "\nBelow is the new question with a preliminary schema\n"}] + problem.to_prompt(including_answer=False) +\
                         [{"type": "text", "text": "Below is the preliminary schema:\n"}, {"type": "text", "text": problem.mental_representation}, {"type": "text", "text": "\nThink about if you could refine the preliminary schema based on the example provided above."}]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": refinement_content},
        {"role": "assistant", "content": schema},
        {"role": "user", "content": "Solve the new question. Select the most appropriate answer."}
    ]

    response = solver_model.interact(
        messages,
        json_format=DynamicResponse,
        temperature=temperature,
        top_p=top_p
    )

    if response is None:
        return None, None, None

    return response, response["final_answer"], messages