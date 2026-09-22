from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.config.settings import ExperimentSettings


class _SchemaConfig(BaseModel):
    prompt: str
    response_cls: type


def _extract_schema_config(settings: ExperimentSettings) -> _SchemaConfig:
    """
    Extract schema-related prompt and response class from ExperimentSettings.

    The legacy pipeline used a global ``config`` module; in the refactored
    version these values are expected to have been injected into
    ``ExperimentSettings.extra`` by the config generation layer.
    """
    extra = settings.extra or {}
    try:
        prompt = extra["SCHEMA_PROMPT"]
        response_cls = extra["SCHEMA_RESPONSE_CLASS"]
    except KeyError as exc:
        raise ValueError(
            "Schema configuration missing in ExperimentSettings.extra. "
            "Expected keys: 'SCHEMA_PROMPT', 'SCHEMA_RESPONSE_CLASS'."
        ) from exc

    return _SchemaConfig(prompt=prompt, response_cls=response_cls)


def LightSchemaActivation(
        problem: Problem,
        knowledges: List[Dict[str, Any]],
        schema_activator_model,
        settings: ExperimentSettings,
) -> Tuple[Any, str]:

    cfg = _extract_schema_config(settings)

    # 1. Prepare the Current Problem (User Message 1)

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
        {"role": "user", "content": refinement_content}
    ]

    # 4. Interact with the Model
    response = schema_activator_model.interact(
        messages, json_format=cfg.response_cls, temperature=0
    )

    if response is None:
        return None, None

    # 5. Dynamic Formatting
    # First, handle 'knowledge_schema' specifically as it has nested structure
    formatted_schema = "\n".join(
        f"##### {key}:\n{value}\n"
        for key, value in response["knowledge_schema"].items()
    )

    # Initialize the parts list with the main Schema
    schema_parts = [f"#### Schema:\n{formatted_schema}"]

    # Iterate over all other keys in the response (e.g., summary, reasoning, etc.)
    for key, value in response.items():
        if key == "knowledge_schema":
            continue

        # Capitalize key for display (e.g., 'summary' -> 'Summary')
        header_name = key.replace("_", " ").capitalize()
        schema_parts.append(f"#### {header_name}:\n{value}")

    # Join all parts with newlines
    schema = "\n".join(schema_parts)

    return response, schema
