import argparse
from typing import Dict
from credentials import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.plugins.solver_registry import SolverResult

url: str = SUPABASE_URL
key: str = SUPABASE_KEY
supabase: Client = create_client(url, key)

import base64
from typing import Any


def truncate_recursively(data, max_length=3000):
    """
    Recursively truncates strings in a dictionary or list if they exceed max_length.
    """
    if isinstance(data, dict):
        return {k: truncate_recursively(v, max_length) for k, v in data.items()}

    elif isinstance(data, list):
        return [truncate_recursively(i, max_length) for i in data]

    elif isinstance(data, str):
        if len(data) > max_length:
            return data[:max_length] + "... [TRUNCATED]"
        return data

    # Return numbers, booleans, None, etc. as is
    return data


def serialize_problem_objects(obj: Any) -> Any:
    """
    Recursively walks through a nested structure (dict or list) and
    converts any 'Problem' instances into a JSON-serializable dictionary.
    """
    # Import here to avoid circular imports if necessary
    from refactored_sa_icl.entity.problems.Problem import Problem

    if isinstance(obj, Problem):
        # 1. Start with the internal dictionary of the Problem instance
        data = vars(obj).copy()

        # 2. Handle the Image Context (Bytes -> Base64)
        if isinstance(obj.context, dict) and 'bytes' in obj.context:
            image_bytes = obj.context['bytes']
            data['context'] = {
                'bytes': base64.b64encode(image_bytes).decode('utf-8'),
                'format': obj.context.get('format', 'png')
            }

        # 3. Recursively clean the dictionary in case it contains other non-serializable objects
        return serialize_problem_objects(data)

    elif isinstance(obj, dict):
        return {k: serialize_problem_objects(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [serialize_problem_objects(i) for i in obj]

    # Return the object as is if it's a basic type (str, int, float, bool, None)
    return obj

def _extract_experiment_attributes(settings: ExperimentSettings) -> dict:
    """
    Convert an ExperimentSettings object into a dict matching the Supabase
    `experiment` table schema. Filters out irrelevant fields automatically.
    """

    return {
        "subject": settings.dataset.subject,
        "knowledge_level": settings.dataset.knowledge_level,
        "solver_type": settings.solver.solver_type,
        "solver_model": settings.model.solver_model,
        "schema_template": settings.solver.schema_template,
        "start_similarity": settings.retrieval.start_similarity,
        "end_similarity": settings.retrieval.end_similarity,
        "num_shots": settings.solver.num_shots,
        "top_k": settings.retrieval.top_k,
        "example_reasoning": settings.solver.with_example_reasoning,
        "exclude": '-'.join(settings.retrieval.exclude),
        "random_from_k": settings.retrieval.random_from_k,
        "use_self_consistency": settings.solver.use_self_consistency,
        "include_answer_in_example": settings.solver.include_answer_in_example,
    }



def get_or_create_experiment(settings: ExperimentSettings):
    """
    Retrieve an existing experiment with matching attributes or create a new one.
    Returns the experiment row (existing or created).
    """

    data = _extract_experiment_attributes(settings)

    # --- 1. Try to find an existing record ---
    query = supabase.table("experiment").select("*")

    for key, value in data.items():
        # JSON columns: 'exclude' must use .eq() also because you match exactly
        query = query.eq(key, value)

    existing = query.execute()

    if existing.data:
        # Found existing experiment
        settings.db_experiment_id = existing.data[0]['id']
        return existing.data[0]

    # --- 2. Create a new experiment ---
    created = (
        supabase
        .table("experiment")
        .insert(data)
        .execute()
    )

    assert len(created.data) == 1, "Failed to create new experiment record."

    settings.db_experiment_id = created.data[0]['id']

    return created.data[0]


import ast


def append_result(settings=ExperimentSettings, solver_results=[SolverResult]):
    """
    Append solver results to the 'solver_result' table in Supabase.
    Each SolverResult is linked to the experiment via experiment_id.
    """

    assert settings.db_experiment_id is not None, "Experiment ID must be set in settings."
    assert len(
        solver_results) == 1, "Only one SolverResult can be appended at a time in this implementation."

    # Extract the single SolverResult
    sr = solver_results[0]

    # --- Helper: Prune based on Length, Density, AND Expand Stringified Lists ---
    def prune_dense_values(data):
        """
        Recursively walks through data.
        1. If a string looks like a list "['a', 'b']", parse and recurse inside.
        2. Otherwise, replace string with 'TOO_LONG_OMMITED' ONLY IF:
           - Total length > 3000 characters AND (Char/Word) ratio > 20
        """
        # 1. Handle Dictionaries
        if isinstance(data, dict):
            return {k: prune_dense_values(v) for k, v in data.items()}

        # 2. Handle Lists
        elif isinstance(data, list):
            return [prune_dense_values(i) for i in data]

        # 3. Handle Strings
        elif isinstance(data, str):
            # A. Check if this string is actually a list in disguise
            stripped = data.strip()
            if len(stripped) > 2 and stripped.startswith('[') and stripped.endswith(']'):
                try:
                    # Safely evaluate string containing Python literals
                    parsed = ast.literal_eval(data)
                    if isinstance(parsed, list):
                        # It was a list! Recurse on the parsed contents
                        return [prune_dense_values(i) for i in parsed]
                except (ValueError, SyntaxError):
                    # If parsing fails, just treat it as a normal string
                    pass

            # B. Standard Length + Density Check
            if len(data) <= 3000:
                return data

            words = data.split()
            if len(words) == 0:
                return "TOO_LONG_OMMITED"

            ratio = len(data) / len(words)

            if ratio > 20:
                return "TOO_LONG_OMMITED"

            return data

        # 4. Handle Bytes
        elif isinstance(data, bytes):
            if len(data) > 3000:
                return "TOO_LONG_OMMITED"
            return data

        return data

    # ------------------------------------------------------------------

    # Sanitize the log
    sanitized_log = prune_dense_values(sr.extra_log)

    # Prepare the record for insertion
    row = {
        "pid": sr.extra_log.get('problem_id'),
        "eid": settings.db_experiment_id,
        "correctness": sr.correct,
        "exact_log": serialize_problem_objects(sanitized_log),
    }

    # Insert one row
    response = (
        supabase
        .table("result")
        .insert(row)
        .execute()
    )

    return response.data[0]


if __name__ == "__main__":
    test_args = argparse.Namespace(
        target_iterations=6,
        subject="GPQA",
        knowledge_level="rag_rerank",
        solver_type="One-Shot + Schema",
        solver_model="GPT-4o Mini",
        schema_template="ChemistrySchema",
        start_similarity=0.1,
        end_similarity=0.8,
        random_from_k=False,
        num_shots=2,
        top_k=1,
        example_reasoning=False,
        reattempts=3,
        exclude=''
    )
    from refactored_sa_icl.config.settings import ExperimentSettings, build_experiment_settings_from_args
    from refactored_sa_icl.pipeline.experiment_runner import load_dataset_and_knowledge_base
    from validation import parse_args

    args = parse_args() if test_args is None else test_args

    # Bridge CLI to immutable ExperimentSettings.
    settings: ExperimentSettings = build_experiment_settings_from_args(args)

    problems, knowledge_base = load_dataset_and_knowledge_base(settings)
    print(get_or_create_experiment(settings))


def load_existing_iterations_from_db(settings: ExperimentSettings) -> Dict[str, int]:
    """
    Query Supabase to determine how many iterations already exist per problem_id
    for the given experiment configuration.

    Returns:
        Dict[str, int]: Mapping of { problem_id (pid) → count_of_existing_rows }.
    """

    # --- 1. Get or verify experiment ID ---
    # If settings.db_experiment_id is not set, attempt to look it up.
    # (Reuse the same attribute-matching logic as get_or_create_experiment)
    if settings.db_experiment_id is None:
        # Import here to avoid circular dependencies.
        from dao import get_or_create_experiment
        exp = get_or_create_experiment(settings)
        settings.db_experiment_id = exp["id"]

    exp_id = settings.db_experiment_id

    # --- 2. Query the result table for all rows matching this experiment ---
    response = (
        supabase
        .table("result")
        .select("pid")
        .eq("eid", exp_id)
        .execute()
    )

    if not response.data:
        return {}

    # --- 3. Count rows per pid ---
    counts: Dict[str, int] = {}
    for row in response.data:
        pid = row["pid"]
        counts[pid] = counts.get(pid, 0) + 1

    return counts
