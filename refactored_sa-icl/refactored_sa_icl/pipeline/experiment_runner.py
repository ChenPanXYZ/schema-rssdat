from __future__ import annotations

from typing import Any, List, Tuple

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
# from refactored_sa_icl.entity.embedder.STEmbedder import STEmbedder
from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.plugins.retrieval_registry import (
    RetrievalResult,
    get_retriever,
)
from refactored_sa_icl.plugins.solver_registry import SolverResult, get_solver
from refactored_sa_icl.usecase.datasets.controller.GenerateMentalRepresentationController import (
    GenerateMentalRepresentationController,
)
from refactored_sa_icl.usecase.datasets.controller.LoadingKnowledgeBaseController import (
    LoadingKnowledgeBaseController,
)
from refactored_sa_icl.usecase.datasets.controller.LoadingProblemsController import (
    LoadingProblemsController,
)


def _initialise_models(settings: ExperimentSettings) -> Tuple[Any, Any, Any]:
    """
    Dynamically import and instantiate the LM classes specified in settings.

    This mirrors the behaviour of the legacy config generator but is scoped to
    the refactored package.
    """
    import importlib

    model_cfg = settings.model

    def _load(model_name: str) -> Any:
        module = importlib.import_module(
            f"refactored_sa_icl.entity.models.{model_name}"
        )
        cls = getattr(module, model_name)
        return cls(ctx_len=model_cfg.context_length)

    memory_model = _load(model_cfg.memory_model)
    solver_model = _load(model_cfg.solver_model)
    schema_activator_model = _load(model_cfg.schema_activator_model)

    return memory_model, solver_model, schema_activator_model




def load_dataset_and_knowledge_base(
    settings: ExperimentSettings,
) -> Tuple[List[Problem], KnowledgeBase]:
    """
    Load problems and knowledge base according to the experiment settings.
    """
    dataset_cfg = settings.dataset

    dataset_names_list = dataset_cfg.dataset_names[0]
    knowledge_base = LoadingKnowledgeBaseController(dataset_names_list)
    problems = LoadingProblemsController(
        dataset_cfg.problem_names_sizes[0][0],
        dataset_cfg.problem_names_sizes[0][1],
    )

    # Now, build the original_synthetic_mappings.
    original_synthetic_mappings = {}
    for problem in knowledge_base.knowledges + problems:
        original_question = problem.reference_to
        reference_type = problem.reference_type

        if original_question not in original_synthetic_mappings:
            original_synthetic_mappings[original_question] = {}

        assert reference_type not in original_synthetic_mappings[original_question] or original_synthetic_mappings[original_question][reference_type] == problem.id, "Duplicate reference type for the same original question."

        original_synthetic_mappings[original_question][reference_type] = problem.id


    settings.dataset.original_synthetic_mappings = original_synthetic_mappings


    return problems, knowledge_base


def prepare_embeddings(
    problems: List[Problem],
    knowledge_base: KnowledgeBase,
    settings: ExperimentSettings,
) -> None:
    """
    Generate mental representations and embeddings for problems and knowledge.
    """
    memory_model, _, _ = _initialise_models(settings)
    embedder = STEmbedder(settings.dataset.embedding_model)

    # Problems
    problems_with_repr = GenerateMentalRepresentationController(
        problems,
        settings.dataset.memory_type,
        memory_model,
        including_answer=False,
        embedder=embedder,
        settings=settings,
    )

    # Knowledge base entries
    knowledge_base.knowledges = GenerateMentalRepresentationController(
        knowledge_base.knowledges,
        settings.dataset.memory_type,
        memory_model,
        including_answer=settings.dataset.memory_including_answer,
        embedder=embedder,
        settings=settings,
    )

    knowledge_base.add_embeddings()

def is_setting_schema_needed(settings: ExperimentSettings) -> bool:
    if settings.solver.solver_type not in ["OneShotSolver", "CoTSolver", "BaselineSolver", "Baseline"] or settings.solver.with_example_reasoning:
        return True


def run_single_problem(
    problem: Problem,
    knowledge_base: KnowledgeBase,
    settings: ExperimentSettings,
    memory_model: Any,
    solver_model: Any,
    schema_activator_model: Any,
) -> SolverResult:
    """
    Run the full retrieval + solver stack for a single problem.
    """
    retriever = get_retriever()
    solver = get_solver()
    # embedder = STEmbedder(settings.dataset.embedding_model)
    embedder = None

    if settings.retrieval.approach == 'rag_rerank':
        GenerateMentalRepresentationController(
            problems=[problem],
            memory_type=settings.dataset.memory_type,
            memory_model=memory_model,
            including_answer=False,
            embedder=embedder,
            settings=settings,
        )

        GenerateMentalRepresentationController(
            problems=knowledge_base.knowledges,
            memory_type=settings.dataset.memory_type,
            memory_model=memory_model,
            including_answer=settings.dataset.memory_including_answer,
            embedder=embedder,
            settings=settings,
        )

    retrieval_result: RetrievalResult = retriever(
        problem=problem,
        knowledge_base=knowledge_base,
        settings=settings,
        solver_model=solver_model # for internal, use the same model as solver to generate internal example.
    )
    if is_setting_schema_needed(settings):
        # only run the following when schema is needed.
        if problem.mental_representation is None:
            # need to generate mental representation on the fly.
            # call the controller
            GenerateMentalRepresentationController(
                problems=[problem],
                memory_type=settings.dataset.memory_type,
                memory_model=memory_model,
                including_answer=False,
                embedder=embedder,
                settings=settings,
            )

        for knowledge in retrieval_result.knowledges:
            if knowledge.mental_representation is None:
                # need to generate mental representation on the fly.
                # call the controller
                GenerateMentalRepresentationController(
                    problems=[knowledge],
                    memory_type=settings.dataset.memory_type,
                    memory_model=memory_model,
                    including_answer=settings.dataset.memory_including_answer,
                    embedder=embedder,
                    settings=settings,
                )

    solver_result: SolverResult = solver(
        problem=problem,
        knowledge_base=knowledge_base,
        settings=settings,
        solver_model=solver_model,
        schema_activator_model=schema_activator_model,
        knowledges=retrieval_result.knowledges,
        similarity=retrieval_result.similarity,
    )

    return solver_result
