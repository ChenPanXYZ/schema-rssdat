from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from .mappings import (
    subject_to_config,
    knowledge_level_to_config,
    solver_type_to_config,
    solver_model_to_config,
)


class ModelSettings(BaseModel):
    """Model-related configuration (solver, memory, schema activator, etc.)."""

    solver_model: str
    memory_model: str
    schema_activator_model: str
    schema_generator_model: str
    context_length: int = 4096


class RetrievalSettings(BaseModel):
    """Knowledge retrieval configuration."""

    approach: str
    top_k: int = 1
    start_similarity: float = 0.0
    end_similarity: float = 1.0
    exclude_self: bool = True
    random_from_k: bool = False
    mapping_path: Optional[str] = None
    exclude: tuple[str, ...] = ("self")
    random_from_k: bool = False


class SolverSettings(BaseModel):
    """Solver behavior configuration."""

    solver_type: str
    schema_template: str
    num_shots: int = 1
    with_example_reasoning: bool = False
    fallback_mode: bool = False
    schema_activator_type: str = "NormalSchemaActivator"
    past_knowledge_mode: str = "question+answer"
    use_self_consistency: bool = False
    sc_samples: int = 5
    sc_temperature: float = 0.7
    sc_top_p: float = 1.0
    include_answer_in_example: bool


class DatasetSettings(BaseModel):
    """Dataset, subject and run configuration."""

    subject: str
    dataset_names: List[List[str]]
    problem_names_sizes: List[List[object]]
    knowledge_level: str
    repeat_num: int = 1
    shuffle: bool = True
    embedding_model: str = "Alibaba-NLP/gte-base-en-v1.5"
    memory_type: str = "semantic"
    memory_including_answer: bool = True
    # Retrieval-related fields injected from mappings; kept here for
    # convenience so that _build_retrieval_settings can read from a single
    # object.
    retreiver_approach: Optional[str] = None
    top_k: int = 1
    start_similarity: float = 0.0
    end_similarity: float = 1.0
    exclude_self: bool = True
    random_from_k: bool = False
    mapping_path: Optional[str] = None
    original_synthetic_mappings: Optional[str] = None
    exclude : tuple[str, ...] = ("self")
    random_from_k: bool = False


class RetrySettings(BaseModel):
    """Error handling / retry configuration."""

    max_reattempts: int = 0


class ExperimentSettings(BaseModel):
    """
    Immutable configuration for a single experiment run.

    This object replaces the dynamic dict-based config used in the legacy
    pipeline. It is constructed once from CLI arguments and mappings and
    then passed down through the pipeline without mutation.
    """

    model_config = ConfigDict(frozen=False) # TODO: Pan: I made it not frozen so I can inject the db id.

    model: ModelSettings
    retrieval: RetrievalSettings
    solver: SolverSettings
    dataset: DatasetSettings
    retry: RetrySettings = Field(default_factory=RetrySettings)

    extra: Dict[str, Any] = Field(default_factory=dict)

    db_experiment_id: Optional[int] = None


def _build_model_settings(solver_model_label: str, memory_model: str) -> ModelSettings:
    """
    Resolve model class names from CLI solver model label using the
    ``solver_model_to_config`` mapping.
    """
    try:
        model_class_name = solver_model_to_config[solver_model_label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown solver_model label '{solver_model_label}'. "
            f"Valid options: {list(solver_model_to_config.keys())}"
        ) from exc

    # In the legacy config, memory_model is always GPT4oMini regardless of the
    # chosen solver model. We keep that behaviour here for parity.
    return ModelSettings(
        solver_model=model_class_name,
        schema_activator_model=model_class_name,
        schema_generator_model=model_class_name,
        memory_model=memory_model,
    )


def _build_dataset_settings(
    subject: str,
    knowledge_level: str,
    repeat_num: int,
    exclude: str,
    start_similarity: float,
    end_similarity: float,
    random_from_k: bool = False,
) -> DatasetSettings:
    """
    Merge subject and knowledge-level specific config into a DatasetSettings
    instance.
    """
    try:
        subject_cfg = subject_to_config[subject]
    except KeyError as exc:
        raise ValueError(
            f"Unknown subject '{subject}'. Valid options: {list(subject_to_config.keys())}"
        ) from exc

    try:
        knowledge_cfg = knowledge_level_to_config[knowledge_level]
    except KeyError as exc:
        raise ValueError(
            "Unknown knowledge_level "
            f"'{knowledge_level}'. Valid options: {list(knowledge_level_to_config.keys())}"
        ) from exc

    # DatasetSettings will also be used as input to RetrievalSettings, so we
    # keep the raw mapping_path/top_k information here as part of dataset.
    mapping = {
        "s": "new_question",
        "d": "new_question_exam",
        "e": "paraphrase"
    }

    # Always start with "self"
    result = ["self"]
    # Add mapped values for each character in the string
    for char in exclude:
        if char in mapping:
            result.append(mapping[char])

    # Convert to tuple
    result = tuple(result)

    merged: Dict[str, Any] = {
        "subject": subject,
        "knowledge_level": knowledge_level,
        "repeat_num": repeat_num,
        "exclude": result,
        "start_similarity": start_similarity,
        "end_similarity": end_similarity,
        "random_from_k": random_from_k
    }
    merged.update(subject_cfg)
    merged.update(knowledge_cfg)

    return DatasetSettings(**merged)  # type: ignore[arg-type]


def _build_retrieval_settings(dataset: DatasetSettings) -> RetrievalSettings:
    """
    Construct RetrievalSettings from the merged dataset configuration.
    """
    approach = getattr(dataset, "retreiver_approach", None)
    if approach is None:
        raise ValueError(
            "retreiver_approach missing in dataset configuration – "
            "check knowledge_level_to_config for this knowledge_level."
        )

    # The legacy config uses 'retreiver_approach' (typo) – we keep the field
    # name for backwards compatibility but expose it here as 'approach'.
    data: Dict[str, Any] = {
        "approach": approach,
        "top_k": getattr(dataset, "top_k", 1),
        "exclude_self": getattr(dataset, "exclude_self", True),
        "random_from_k": getattr(dataset, "random_from_k", False),
        "mapping_path": getattr(dataset, "mapping_path", None),
        "exclude": getattr(dataset, "exclude", ("self")),
        "start_similarity": dataset.start_similarity,
        "end_similarity": dataset.end_similarity,
        "random_from_k": dataset.random_from_k,
    }
    return RetrievalSettings(**data)


def _build_solver_settings(
    solver_type_label: str,
    schema_template: str,
    num_shots: int,
    with_example_reasoning: bool,
    use_self_consistency: bool,
    include_answer_in_example: bool
) -> SolverSettings:
    """
    Build SolverSettings from a human-friendly solver_type label plus
    schema/shot configuration.
    """
    try:
        base = solver_type_to_config[solver_type_label].copy()
    except KeyError as exc:
        raise ValueError(
            f"Unknown solver_type label '{solver_type_label}'. "
            f"Valid options: {list(solver_type_to_config.keys())}"
        ) from exc

    # solver_type_to_config may override top_k / num_shots; we respect that,
    # but allow the explicit CLI num_shots to override if provided > 0.
    if num_shots > 0:
        base["num_shots"] = num_shots

    base.setdefault("num_shots", 1)
    base.setdefault("fallback_mode", False)
    base.setdefault("schema_activator_type", "NormalSchemaActivator")
    base.setdefault("past_knowledge_mode", "question+answer")

    if use_self_consistency:
        base["use_self_consistency"] = use_self_consistency
    else:
        base["use_self_consistency"] = False


    base["schema_template"] = schema_template
    base["with_example_reasoning"] = with_example_reasoning
    base["include_answer_in_example"] = include_answer_in_example

    return SolverSettings(**base)  # type: ignore[arg-type]


def _load_schema_template(schema_template: str) -> Dict[str, Any]:
    """
    Import the schema template module and extract prompt/response artefacts.

    This mirrors the behaviour of the legacy config generator but stores the
    results inside ExperimentSettings.extra instead of a global config dict.
    """
    import importlib

    try:
        module = importlib.import_module(
            f"refactored_sa_icl.entity.schema_templates.{schema_template}"
        )
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown schema_template '{schema_template}'. "
            "Expected a module under "
            "'refactored_sa_icl.entity.schema_templates'."
        ) from exc

    try:
        Response = getattr(module, "Response")
        schema_prompt = getattr(module, "SCHEMA_PROMPT")
        sample_question = getattr(module, "SCHEMA_SAMPLE_QUESTION")
        sample_response = getattr(module, "SCHEMA_SAMPLE_RESPONSE")
        schema_solver_prompt = getattr(module, "SCHEMA_SOLVER_PROMPT")
        # schema_refinement_prompt = getattr(module, "SCHEMA_REFINEMENT_PROMPT")
    except AttributeError as exc:
        raise ValueError(
            "Schema template module is missing one or more required "
            "attributes: 'Response', 'SCHEMA_PROMPT', "
            "'SCHEMA_SAMPLE_QUESTION', 'SCHEMA_SAMPLE_RESPONSE', "
            "'SCHEMA_SOLVER_PROMPT', "
            "'SCHEMA_REFINEMENT_PROMPT"
        ) from exc

    return {
        "SCHEMA_RESPONSE_CLASS": Response,
        "SCHEMA_PROMPT": schema_prompt,
        "SCHEMA_SAMPLE_QUESTION": sample_question,
        "SCHEMA_SAMPLE_RESPONSE": sample_response,
        "SCHEMA_SOLVER_PROMPT": schema_solver_prompt,
        # "SCHEMA_REFINEMENT_PROMPT": schema_refinement_prompt,
    }


def build_experiment_settings_from_args(args: Any) -> ExperimentSettings:
    """
    Build an immutable ExperimentSettings instance from an argparse-like
    object (e.g. ``argparse.Namespace``) using our static mappings.

    The expected attributes on ``args`` mirror the legacy validation.py
    CLI: subject, knowledge_level, solver_type, solver_model,
    schema_template, num_shots, target_iterations (or repeat_num),
    example_reasoning, and reattempts.
    """
    try:
        subject: str = args.subject
        knowledge_level: str = args.knowledge_level
        solver_type_label: str = args.solver_type
        solver_model_label: str = args.solver_model
        schema_template: str = args.schema_template
        memory_model: str = args.memory_model
    except AttributeError as exc:
        raise ValueError(
            "Missing required CLI arguments on 'args'. "
            "Expected at least: subject, knowledge_level, "
            "solver_type, solver_model, schema_template."
        ) from exc

    repeat_num: int = getattr(args, "target_iterations", getattr(args, "repeat_num", 1))
    num_shots: int = getattr(args, "num_shots", 1)
    with_example_reasoning: bool = getattr(args, "example_reasoning", False)
    max_reattempts: int = getattr(args, "reattempts", 0)
    exclude: str = getattr(args, "exclude", "")
    start_similarity: float = getattr(args, "start_similarity", 0.0)
    end_similarity: float = getattr(args, "end_similarity", 1.0)
    use_self_consistency: bool = getattr(args, "use_self_consistency", False)
    include_answer_in_example: bool = getattr(args, "include_answer_in_example", True)

    dataset = _build_dataset_settings(
        subject=subject,
        knowledge_level=knowledge_level,
        repeat_num=repeat_num,
        exclude=exclude,
        start_similarity=start_similarity,
        end_similarity=end_similarity,
        random_from_k=getattr(args, "random_from_k", False),
    )
    retrieval = _build_retrieval_settings(dataset)
    solver = _build_solver_settings(
        solver_type_label=solver_type_label,
        schema_template=schema_template,
        num_shots=num_shots,
        with_example_reasoning=with_example_reasoning,
        use_self_consistency=use_self_consistency,
        include_answer_in_example=include_answer_in_example
    )
    model = _build_model_settings(solver_model_label=solver_model_label, memory_model=memory_model)
    retry = RetrySettings(max_reattempts=max_reattempts)

    # Load schema-template-specific artefacts into the extra dict so that
    # downstream components (mental representation generator, schema
    # activator, schema-aware solvers) can access them without globals.
    extra: Dict[str, Any] = _load_schema_template(schema_template)

    return ExperimentSettings(
        model=model,
        retrieval=retrieval,
        solver=solver,
        dataset=dataset,
        retry=retry,
        extra=extra,
    )
