from __future__ import annotations

import base64
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

import pandas as pd

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.plugins.solver_registry import SolverResult


@dataclass
class ResultRow:
    process_id: str
    problem_id: str
    correctness: int
    subject: str
    knowledge_base: str
    knowledge_level: str
    solver_type: str
    solver_model: str
    schema_template: str
    with_example_reasoning: bool


def _encode_string_to_id(input_string: str) -> str:
    return base64.urlsafe_b64encode(input_string.encode()).decode()


def build_results_file_path(settings: ExperimentSettings) -> str:
    """
    Reproduce the legacy naming convention used by validation.py results.
    """
    subject = settings.dataset.subject
    knowledge_level = settings.dataset.knowledge_level
    solver_type = settings.solver.solver_type
    solver_model = settings.model.solver_model
    schema_template = settings.solver.schema_template
    with_example_reasoning = settings.solver.with_example_reasoning

    filename = (
        f"{subject}_{knowledge_level}_{solver_type}_"
        f"{solver_model}_{schema_template}_{with_example_reasoning}.csv"
    )
    return os.path.join("subsets", filename)


def build_rows(
    settings: ExperimentSettings, solver_results: Iterable[SolverResult]
) -> List[ResultRow]:
    rows: List[ResultRow] = []
    for result in solver_results:
        log = result.extra_log
        if log.get("correct") is None:
            continue

        problem_id = log.get("problem_id", "")
        correctness = int(bool(log["correct"]))

        rows.append(
            ResultRow(
                process_id=_encode_string_to_id(str(log)),
                problem_id=problem_id,
                correctness=correctness,
                subject=settings.dataset.subject,
                knowledge_base="dense",
                knowledge_level=settings.dataset.knowledge_level,
                solver_type=settings.solver.solver_type,
                solver_model=settings.model.solver_model,
                schema_template=settings.solver.schema_template,
                with_example_reasoning=settings.solver.with_example_reasoning,
            )
        )
    return rows


def append_results(
    settings: ExperimentSettings,
    solver_results: Iterable[SolverResult],
    results_file: Optional[str] = None,
) -> None:
    """
    Append experiment results to the CSV file, creating it with headers if
    required.

    If ``results_file`` is provided, results are written to that path.
    Otherwise, the path is derived from ``settings`` using
    :func:`build_results_file_path`.
    """
    rows = build_rows(settings, solver_results)
    if not rows:
        return

    if results_file is None:
        results_file = build_results_file_path(settings)

    print(results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    new_df = pd.DataFrame([asdict(row) for row in rows])

    try:
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            with open(results_file, "r+b") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            new_df.to_csv(results_file, mode="a", header=False, index=False)
        else:
            new_df.to_csv(results_file, mode="w", header=True, index=False)
    except Exception:
        import traceback

        traceback.print_exc()
        raise



