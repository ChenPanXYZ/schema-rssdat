import os
from collections import Counter
from typing import List

import pandas as pd

from refactored_sa_icl.config.settings import ExperimentSettings


def apply_self_consistency(
    run_once_fn,
    *,
    problem,
    schema,
    solver_model,
    knowledges,
    formatted_knowledges,
    settings,
):
    """
    SC wrapper: samples reasoning multiple times and majority-votes the final answer.
    """

    all_responses = []
    final_answers = []
    all_meesages =[]
    for _ in range(settings.solver.sc_samples):
        response, ans, messages = run_once_fn(
            problem,
            schema,
            solver_model,
            knowledges,
            formatted_knowledges,
            settings,
            temperature=settings.solver.sc_temperature,   # VERY IMPORTANT
            top_p=settings.solver.sc_top_p,
        )

        all_responses.append(response)
        final_answers.append(ans)
        all_meesages.extend(all_meesages)

    # majority vote
    counts = Counter(final_answers)
    majority_answer = counts.most_common(1)[0][0]

    if settings.solver.sc_samples:
        return {
            "samples": all_responses,
            "votes": final_answers,
            "counts": counts,
            "final": majority_answer,
        }, majority_answer, all_meesages
    else:
        return None, majority_answer, all_meesages



def _load_reasoning_for_id(example_problem_id: str) -> str | None:
    """
    Look up the example reasoning by question_id in any *_reasoning_mapping.csv at repo root.
    Falls back to None if not found.
    """
    # Project root: go up three levels from this file
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../reasoning_mappings"))
    try_paths = []
    try:
        for fname in os.listdir(root_dir):
            if fname.endswith("_reasoning_mapping.csv"):
                try_paths.append(os.path.join(root_dir, fname))
    except Exception:
        # If listing fails, just return None

        return None

    for csv_path in try_paths:
        try:
            df = pd.read_csv(csv_path)
            if "question_id" in df.columns and "reasoning_path" in df.columns:
                rows = df[df["question_id"].astype(str) == str(example_problem_id)]
                if not rows.empty:
                    print(f"Loaded reasoning mapping from {csv_path} for question {example_problem_id}")
                    return str(rows.iloc[0]["reasoning_path"])
        except Exception:
            continue
    return None


def get_role_playing_text(settings: ExperimentSettings) -> str:
    schema_template = settings.solver.schema_template
    if schema_template == "MathematicsSchema":
        return "You are an expert in mathematics."
    elif schema_template == "BusinessSchema":
        return "You are an expert in business."

    elif schema_template == "ChemistrySchema":
        return "" #TODO note that ChemistrySchema is actually the general one.
    else:
        return ""




def get_one_shot_system_prompt(include_answer_in_example: bool, with_example_reasoning: bool) -> str:
    parts = []
    if include_answer_in_example:
        parts.append("solution")
    if with_example_reasoning:
        parts.append("reasoning")

    # 2. Join them gracefully
    if len(parts) > 1:
        # Joins everything with a comma, except the last item which gets "and"
        extras = ", ".join(parts[:-1]) + " and " + parts[-1]
    else:
        extras = parts[0]

    prompt = f"You are given an example question along with its {extras}. Then, select the most appropriate answer for a **new question**."
    return prompt


def get_schema_system_prompt(include_answer_in_example):
    if include_answer_in_example:
        return "You will be givin an example question along with its solution and schema. Then, you will be given a new question with a preliminary schema. Refine the preliminary schema by incorporting the relevant part from the example's schema. Then, select the most appropriate answer for the new question."
    else:
        return "You will be givin an example question along with its schema. Then, you will be given a new question with a preliminary schema. Refine the preliminary schema by incorporting the relevant part from the example's schema. Then, select the most appropriate answer for the new question."



def get_schema_refinement_prompt(include_answer_in_example,knowledges,problem) -> List:
    if include_answer_in_example:
        first = 'Below is an example question along with its solution and schema'
    else:
        first = 'Below is an example question along with its schema'

    refinement_content = []
    refinement_content.append({"type": "text", "text": first + ":\n"})




    refinement_content += knowledges + [
        {"type": "text", "text": "\nBelow is the **new question** with a preliminary schema\n"}] + problem.to_prompt(including_answer=False) +\
                         [{"type": "text", "text": "Below is the preliminary schema:\n"}, {"type": "text", "text": problem.mental_representation}, {"type": "text", "text": "Activate the schema by rewriting the preliminary schema based on the example question and its schema."}]

    return refinement_content
