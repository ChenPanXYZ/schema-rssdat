import argparse
import traceback
from typing import Any

from refactored_sa_icl.config.settings import (
    ExperimentSettings,
    build_experiment_settings_from_args,
)
from refactored_sa_icl.dao.dao import get_or_create_experiment
from refactored_sa_icl.logging_utils.results_writer import build_results_file_path
from refactored_sa_icl.pipeline.experiment_runner import (
    load_dataset_and_knowledge_base,
    prepare_embeddings,
)
from refactored_sa_icl.pipeline.problem_runner import run_problems


def str2bool(v: str | bool) -> bool:
    """
    Convert string input to boolean.
    Handles common variations of 'true' and 'false'.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (e.g., True/False, 1/0, Yes/No).')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refactored validation interface for Schema-ICL."
    )

    # ... (Other arguments remain the same)

    parser.add_argument(
        "--target_iterations",
        type=int,
        required=True,
        help="Target number of iterations per problem.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        choices=list(
            build_experiment_settings_from_args.__globals__[
                "subject_to_config"
            ].keys()
        ),
        help="Subject / dataset family.",
    )
    parser.add_argument(
        "--knowledge_level",
        type=str,
        required=True,
        choices=list(
            build_experiment_settings_from_args.__globals__[
                "knowledge_level_to_config"
            ].keys()
        ),
        help="Knowledge level (e.g., High, Medium, Low).",
    )
    parser.add_argument(
        "--solver_type",
        type=str,
        required=True,
        choices=list(
            build_experiment_settings_from_args.__globals__[
                "solver_type_to_config"
            ].keys()
        ),
        help="Solver type (Baseline, Schema Only, etc.).",
    )
    parser.add_argument(
        "--solver_model",
        type=str,
        required=True,
        choices=list(
            build_experiment_settings_from_args.__globals__[
                "solver_model_to_config"
            ].keys()
        ),
        help="Logical solver model name (e.g., GPT-4o Mini).",
    )
    parser.add_argument(
        "--schema_template",
        type=str,
        required=True,
        help="Schema template module name.",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=1,
        help="Number of shots for example-based solvers.",
    )

    # --- UPDATED ARGUMENTS ---
    parser.add_argument(
        "--example_reasoning",
        type=str2bool,  # Changed from type=bool
        default=False,
        help="Whether to use examples with CoT reasoning.",
    )
    parser.add_argument(
        "--random_from_k",
        type=str2bool,  # Added this as it appeared in your CLI command
        default=False,
        help="Whether to sample randomly from k.",
    )
    parser.add_argument(
        "--use_self_consistency",
        type=str2bool,  # Added this as it appeared in your CLI command
        default=False,
        help="Whether to use self-consistency.",
    )
    # -------------------------

    parser.add_argument(
        "--reattempts",
        type=int,
        default=20,
        help="Maximum reattempts per problem on failure.",
    )

    # If your command uses --start_similarity, --end_similarity, --top_k, etc.
    # make sure they are added here as well so parse_known_args doesn't dump them into 'unknown'
    parser.add_argument(
        "--start_similarity",
        type=float,
        default=0,
        help="Starting similarity threshold for retrieval.",
    )
    parser.add_argument(
        "--end_similarity",
        type=float,
        default=1.0,
        help="Ending similarity threshold for retrieval.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Top K similar items to retrieve.",
    )

    parser.add_argument(
        "--include_answer_in_example",
        type=bool,
        required=True
    )

    parser.add_argument(
        "--memory_model",
        type=str,
        required=True,
        help="Memory model",
    )

    args, unknown = parser.parse_known_args()
    return args

def exp_need_embeddings(settings: ExperimentSettings) -> bool:
    """
    Determine if the experiment requires embeddings based on the knowledge level.
    """
    return False

def main(raw_args: Any | None = None) -> None:
    try:
        # If raw_args is passed (e.g. from another script), use it.
        # Otherwise, parse from the command line.
        args = raw_args if raw_args is not None else parse_args()
        # Bridge CLI to immutable ExperimentSettings.
        settings: ExperimentSettings = build_experiment_settings_from_args(args)

        problems, knowledge_base = load_dataset_and_knowledge_base(settings)
        if settings.retrieval == "rag_rerank":
            prepare_embeddings(problems, knowledge_base, settings)

        results_file = build_results_file_path(settings)

        get_or_create_experiment(settings)
        assert settings.db_experiment_id is not None, "Experiment ID should be set before running problems."

        run_problems(
            problems=problems,
            knowledge_base=knowledge_base,
            settings=settings,
            results_file=results_file,
        )
    except Exception:
        traceback.print_exc()
        raise

# --- THIS IS THE KEY ADDITION ---
if __name__ == "__main__":
    main()
