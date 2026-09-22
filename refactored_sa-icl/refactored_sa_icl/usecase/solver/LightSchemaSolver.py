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


def LightSchemaSolver(
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
        reasoning: str
        final_answer: AnswerEnum

    assert formatted_knowledges, "formatted_knowledges (multimodal list) must be provided."

    # refinement_intro_text = (
    #     "You are now given an example question along with its schema and solution."
    # )
    #
    # refinement_outtro_text = (
    #     "\nReflect on how you could refine or improve your originally generated schema for the **original** question. "
    #     "Focus on alignment in categories, scope specificity, and consistency of abstraction. "
    #     "For your reference, here is the **original** question again:\n"
    # )
    #
    #
    # # Combine: [Intro Text] + [Past Knowledge Images/Text]
    # refinement_content = [
    #                          {"type": "text", "text": refinement_intro_text}
    #                      ] + formatted_knowledges + [{"type": "text", "text": refinement_outtro_text}] + problem.to_prompt(including_answer=False)
    #
    # # 2. Prepare Current Problem Content (Multimodal)
    # # Replacing str(problem) with proper multimodal prompt to ensure images are included
    # current_problem_content = problem.to_prompt(including_answer=False)
    #
    # schema_solver_prompt = _get_schema_solver_prompt(settings)
    #
    # final_instruction_text = "Now that you've refined your schema, use the refined schema to answer the **original** question.\n"
    # final_prompt = "Select the most appropriate answer. Answer with candidate string only, without index."
    #
    # # 2. Construct the final message content by combining the instruction with the problem list
    # # We put the instruction first, then the actual problem content (text/images)
    # final_content = [{"type": "text", "text": final_instruction_text},
    #                  {"type": "text", "text": final_prompt}]
    #
    #
    # # 3. Construct Conversation History
    # messages = [
    #     {"role": "system", "content": schema_solver_prompt},
    #
    #     # User: Current Problem (Multimodal)
    #     {"role": "user", "content": current_problem_content},
    #
    #     # Assistant: Initial Mental Representation (Text)
    #     {"role": "assistant", "content": problem.mental_representation},
    #
    #     # User: Refinement Instructions + Past Knowledge (Multimodal)
    #     {"role": "user", "content": refinement_content},
    #
    #     # Assistant: Refined Schema (Text - passed in as argument)
    #     {"role": "assistant", "content": schema},
    #
    #     # User: Final Solve Instruction (Text)
    #     {
    #         "role": "user",
    #         "content": final_content
    #     },
    # ]



    # 1. Prepare the Current Problem (User Message 1)
    current_problem_content = problem.to_prompt(including_answer=False)

    current_problem_content.append({"type": "text", "text": "Below is a **preliminary schema**\n"})

    current_problem_content.append({"type": "text", "text": problem.mental_representation})

    # 3. Construct the Conversation History
    system_prompt = (
        "You are given an example question along with its solution and schema. "
        "Then, you will be given a new question. "
        "Your task is to create a schema based on the example provided. "
    )

    # Combine: [Intro] + [Past Knowledge List] + [Outro]
    refinement_content = knowledges + [
        {"type": "text", "text": "\nBelow is the new question.\n"}] + problem.to_prompt(including_answer=False) +\
                         [{"type": "text", "text": "\nCreate a schema based on the example provided above."}]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": refinement_content},
        {"role": "assistant", "content": schema},
        {"role": "user", "content": "Now, use the schema above to answer the question. Select the most appropriate answer. Answer with candidate string only, without index."}
    ]

    response = solver_model.interact(
        messages,
        json_format=DynamicResponse,
        temperature=temperature,
        top_p=top_p
    )

    if response is None:
        return None, None

    return response, response["final_answer"]
