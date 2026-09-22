from __future__ import annotations

import traceback
from argparse import Namespace
from typing import Any, Dict

from .mappings import (
    subject_to_config,
    knowledge_level_to_config,
    solver_type_to_config,
    solver_model_to_config,
)
from .settings import (
    DatasetSettings,
    ExperimentSettings,
    ModelSettings,
    RetrievalSettings,
    RetrySettings,
    SolverSettings,
)


class ExperimentSettingsError(ValueError):
    """Raised when CLI arguments cannot be mapped to a valid ExperimentSettings."""


def _build_dataset_settings(args: Namespace) -> DatasetSettings:
    if args.subject not in subject_to_config:
        msg = f"Unknown subject '{args.subject}'. Available: {list(subject_to_config.keys())}"
        raise ExperimentSettingsError(msg)

    subject_cfg: Dict[str, Any] = subject_to_config[args.subject]

    if args.knowledge_level not in knowledge_level_to_config:
        msg = (
            f"Unknown knowledge_level '{args.knowledge_level}'. "
            f"Available: {list(knowledge_level_to_config.keys())}"
        )
        raise ExperimentSettingsError(msg)

    knowledge_cfg: Dict[str, Any] = knowledge_level_to_config[args.knowledge_level]

    dataset_cfg: Dict[str, Any] = {
        "subject": args.subject,
        "dataset_names": subject_cfg["dataset_names"],
        "problem_names_sizes": subject_cfg["problem_names_sizes"],
        "knowledge_level": args.knowledge_level,
        "repeat_num": args.target_iterations,
    }

    # Merge in keys that DatasetSettings knows about (e.g. top-level defaults are in the
    # dataclass, but subject-specific overrides like embedding_model could go here).
    dataset_cfg.update(
        {
            k: v
            for k, v in knowledge_cfg.items()
            if k in {"top_k"}  # only include fields that DatasetSettings expects
        }
    )

    return DatasetSettings(**dataset_cfg)


def _build_retrieval_settings(args: Namespace) -> RetrievalSettings:
    if args.knowledge_level not in knowledge_level_to_config:
        msg = (
            f"Unknown knowledge_level '{args.knowledge_level}'. "
            f"Available: {list(knowledge_level_to_config.keys())}"
        )
        raise ExperimentSettingsError(msg)

    knowledge_cfg: Dict[str, Any] = knowledge_level_to_config[args.knowledge_level]
    subject_cfg: Dict[str, Any] = subject_to_config[args.subject]

    return RetrievalSettings(
        approach=knowledge_cfg.get("retreiver_approach", "rag_rerank"),
        top_k=knowledge_cfg.get("top_k", 1),
        exclude_self=True,
        random_from_k=False,
        mapping_path=subject_cfg.get("mapping_path"),
    )


def _build_solver_settings(args: Namespace) -> SolverSettings:
    if args.solver_type not in solver_type_to_config:
        msg = (
            f"Unknown solver_type '{args.solver_type}'. "
            f"Available: {list(solver_type_to_config.keys())}"
        )
        raise ExperimentSettingsError(msg)

    solver_type_cfg: Dict[str, Any] = solver_type_to_config[args.solver_type]

    return SolverSettings(
        solver_type=solver_type_cfg["solver_type"],
        schema_template=args.schema_template,
        num_shots=args.num_shots,
        with_example_reasoning=args.example_reasoning,
        fallback_mode=False,
        past_knowledge_mode="question+answer",
        schema_activator_type="NormalSchemaActivator",
    )


def _build_model_settings(args: Namespace) -> ModelSettings:
    if args.solver_model not in solver_model_to_config:
        msg = (
            f"Unknown solver_model '{args.solver_model}'. "
            f"Available: {list(solver_model_to_config.keys())}"
        )
        raise ExperimentSettingsError(msg)

    memory_model = args.memory_model

    model_name = solver_model_to_config[args.solver_model]

    return ModelSettings(
        solver_model=model_name,
        schema_activator_model=model_name,
        schema_generator_model=model_name,
        memory_model=memory_model,
        context_length=4096,
    )


def build_experiment_settings(args: Namespace) -> ExperimentSettings:
    """
    Build an immutable ExperimentSettings instance from CLI arguments.

    This function centralises all mapping logic and provides nicer error
    messages than the legacy dict-based config.
    """
    try:
        dataset = _build_dataset_settings(args)
        retrieval = _build_retrieval_settings(args)
        solver = _build_solver_settings(args)
        model = _build_model_settings(args)
        retry = RetrySettings(max_reattempts=args.reattempts)

        return ExperimentSettings(
            model=model,
            retrieval=retrieval,
            solver=solver,
            dataset=dataset,
            retry=retry,
        )
    except ExperimentSettingsError:
        # Already a user-facing config error, re-raise.
        raise
    except Exception as exc:  # pragma: no cover - defensive
        tb = traceback.format_exc()
        message = (
            f"Failed to build ExperimentSettings from CLI arguments: {exc}\n{tb}"
        )
        raise ExperimentSettingsError(message) from exc
