import json
import os
import pickle
import traceback
from typing import List, Dict, Any

from tqdm import tqdm

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.problems.Problem import Problem
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../")
)

def generate_mental_representation(
        problem: Problem,
        model,
        settings: ExperimentSettings,
        including_answer: bool,
):

    extra = settings.extra or {}
    prompt = extra.get("SCHEMA_PROMPT")
    Response = extra.get("SCHEMA_RESPONSE_CLASS")
    sample_problem = extra.get("SCHEMA_SAMPLE_QUESTION")
    sample_response = extra.get("SCHEMA_SAMPLE_RESPONSE")

    if not all([prompt, Response]):
        raise ValueError(
            "Schema configuration missing in ExperimentSettings.extra."
        )

    # Note: problem.to_prompt returns a list of content blocks for multimodal support
    if sample_problem is not None:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": sample_problem},
            {"role": "assistant", "content": sample_response.model_dump_json()},
            {"role": "user", "content": problem.to_prompt(including_answer)}
        ]
    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": problem.to_prompt(including_answer)}
        ]

    generated_raw = model.interact(
        messages=messages,
        json_format=Response,
        temperature=0.0,
        max_tokens=4096
    )

    # Convert dict to string then validate to ensure we have a clean Response object
    generated_json_str = json.dumps(generated_raw)
    parsed_response = Response.model_validate_json(generated_json_str)

    # We convert the entire object to a dict to handle attributes dynamically
    full_dict = parsed_response.model_dump()

    # Extract schema specifically as it's the core component
    schema = full_dict.pop("knowledge_schema", {})

    # The rest of the keys (summary, reasoning, etc.) are returned as a 'metadata' dict
    return schema, full_dict


import os
import hashlib


# Assuming PROJECT_ROOT is defined elsewhere
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_file_hash(filepath: str) -> str:
    """Calculates the MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files efficiently
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_mental_representation_path(schema_template_name: str) -> str:
    """
    Generates a path for the mental representation based on the
    schema template's source code version.
    """
    # 1. Locate the source schema template file
    # Note: adjusting path to match your specific structure
    schema_source_path = os.path.join(
        PROJECT_ROOT,
        "refactored_sa-icl/refactored_sa_icl/entity/schema_templates",
        f"{schema_template_name}.py"
    )

    if not os.path.exists(schema_source_path):
        raise FileNotFoundError(f"Schema template not found at: {schema_source_path}")

    # 2. Generate deterministic version from file content
    version_hash = get_file_hash(schema_source_path)

    # 3. Construct the final path
    # Structure: data/mental_representations/{schema_name}/{hash}
    # This creates a unique folder for every version of the code
    final_path = os.path.join(
        PROJECT_ROOT,
        "data/mental_representations",
        schema_template_name,
        version_hash
    )

    # Optional: Ensure the directory exists immediately
    os.makedirs(final_path, exist_ok=True)

    return final_path

def GenerateMentalRepresentationController(
        problems: List[Problem],
        memory_type,
        memory_model,
        including_answer,
        embedder,
        settings: ExperimentSettings,
):
    extra = settings.extra or {}
    Response = extra.get("SCHEMA_RESPONSE_CLASS")
    if Response is None:
        raise ValueError("SCHEMA_RESPONSE_CLASS missing in ExperimentSettings.extra")

    schema_template_name = Response.__module__.split(".")[-1]


    mental_representations_path = (
        get_mental_representation_path(schema_template_name) + "/"
        f"{memory_type}_by_{memory_model.__class__.__name__}_including_answer_"
        f"{including_answer}"
    )

    print(f"Path: {mental_representations_path}")

    if memory_type == "semantic":
        for i in tqdm(range(len(problems)), desc="Handling mental representations 🧠"):
            problem = problems[i]
            try:
                if os.path.exists(f'{mental_representations_path}/{problem.id}.json'):
                    with open(f'{mental_representations_path}/{problem.id}.json', "r") as file:
                        mental_representation = json.load(file)
                else:
                    os.makedirs(mental_representations_path, exist_ok=True)

                    # Generate dynamically
                    schema, other_attributes = generate_mental_representation(
                        problem, memory_model, settings, including_answer=including_answer
                    )

                    # Combine into a single dictionary for storage
                    mental_representation = {
                        "knowledge_schema": schema,
                        **other_attributes  # Flattens summary, reasoning, etc. into top level
                    }

                    with open(f'{mental_representations_path}/{problem.id}.json', "w") as file:
                        json.dump(mental_representation, file)

                # --- DYNAMIC STRING FORMATTING ---

                # 1. Format the 'knowledge_schema' dictionary
                schema_dict = mental_representation.get("knowledge_schema", {})
                formatted_schema = "\n".join(
                    f"##### {key}:\n{value}\n" for key, value in schema_dict.items()
                )

                representation_parts = [f"#### Schema:\n{formatted_schema}"]

                # 2. Add all other top-level keys dynamically (Summary, Reasoning, etc.)
                for key, value in mental_representation.items():
                    if key == "knowledge_schema" or value is None:
                        continue

                    # Clean header: "reasoning_trace" -> "Reasoning trace"
                    header_name = key.replace("_", " ").capitalize()
                    representation_parts.append(f"#### {header_name}:\n{value}")

                # 3. Finalize the mental_representation string on the Problem object
                problem.mental_representation = "\n".join(representation_parts)

            except Exception as e:
                print(f"Error processing row {i} (ID: {problem.id}): {e}")
                traceback.print_exc()
                continue

        return problems

    elif memory_type == "episode":
        return None, None
