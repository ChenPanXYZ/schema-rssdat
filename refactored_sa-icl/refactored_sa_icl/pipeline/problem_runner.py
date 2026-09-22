from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import pandas as pd

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.dao.dao import append_result, load_existing_iterations_from_db
from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.logging_utils.results_writer import (
    append_results,
    build_results_file_path,
)
from refactored_sa_icl.pipeline.experiment_runner import (
    _initialise_models,
    run_single_problem,
)
from refactored_sa_icl.plugins.solver_registry import SolverResult
from collections import deque

def _load_existing_iterations(
    results_file: str, settings: ExperimentSettings
) -> Dict[str, int]:
    """
    Load the number of completed iterations per problem from an existing
    results CSV.

    The count is restricted to rows that match the current experiment
    configuration (subject, knowledge level, solver type/model,
    schema_template, with_example_reasoning, and a dense knowledge base).
    """
    if not os.path.exists(results_file):
        return {}

    try:
        df = pd.read_csv(results_file)
    except Exception:
        import traceback

        traceback.print_exc()
        return {}

    if df.empty:
        return {}

    # Older result files may not have all the newer columns; we guard access
    # and fall back to an empty mapping in that case.
    required_columns = {
        "problem_id",
        "process_id",
        "subject",
        "knowledge_base",
        "knowledge_level",
        "solver_type",
        "solver_model",
        "schema_template",
        "with_example_reasoning",
        "use_self_consistency",
    }
    if not required_columns.issubset(df.columns):
        return {}

    mask = (
        (df["subject"] == settings.dataset.subject)
        & (df["knowledge_base"] == "dense")
        & (df["knowledge_level"] == settings.dataset.knowledge_level)
        & (df["solver_type"] == settings.solver.solver_type)
        & (df["solver_model"] == settings.model.solver_model)
        & (df["schema_template"] == settings.solver.schema_template)
        & (df["with_example_reasoning"] == settings.solver.with_example_reasoning)
        & (df["use_self_consistency"] == settings.solver.use_self_consistency)
    )

    filtered = df[mask]
    if filtered.empty:
        return {}

    grouped = filtered.groupby("problem_id")["process_id"].size()
    return grouped.to_dict()


def run_problems(
    problems: Iterable[Problem],
    knowledge_base: KnowledgeBase,
    settings: ExperimentSettings,
    results_file: Optional[str] = None,
) -> List[SolverResult]:
    """
    Run all problems up to the requested number of iterations, appending
    results to the CSV as we go.

    For each problem we:
    - Look up how many iterations have already been recorded in the results
      CSV for the current configuration.
    - Only run the remaining iterations needed to reach
      ``settings.dataset.repeat_num``.
    - Append every new SolverResult to the CSV immediately.
    """
    if results_file is None:
        results_file = build_results_file_path(settings)

    memory_model, solver_model, schema_activator_model = _initialise_models(
        settings
    )

    existing_iterations = load_existing_iterations_from_db(settings)


    results: List[SolverResult] = []
    target_iterations = settings.dataset.repeat_num

    for problem in problems:
        problem_id = getattr(problem, "id", None)
        if problem_id is None:
            # If we cannot identify the problem, fall back to always running
            # the full number of iterations.
            completed_iterations = 0
        else:
            completed_iterations = int(existing_iterations.get(str(problem_id), 0))

        if completed_iterations >= target_iterations:
            # This problem has already reached (or exceeded) the target number
            # of iterations; skip it.
            continue

        error_count = 0
        iteration = completed_iterations

        while iteration < target_iterations:
            try:
                solver_result = run_single_problem(
                    problem=problem,
                    knowledge_base=knowledge_base,
                    settings=settings,
                    memory_model=memory_model,
                    solver_model=solver_model,
                    schema_activator_model=schema_activator_model,
                )
                results.append(solver_result)
                append_results(
                    settings=settings,
                    solver_results=[solver_result],
                    results_file=results_file,
                )
                append_result(settings=settings, solver_results=[solver_result])
                iteration += 1
                print(solver_result.correct)
            except Exception:
                import traceback

                error_count += 1
                traceback.print_exc()
                if error_count >= settings.retry.max_reattempts:
                    break
                # otherwise, retry the same iteration

    return results
