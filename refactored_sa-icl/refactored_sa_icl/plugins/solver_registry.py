from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.usecase.datasets.controller.FormatPastKnowledgeController import (
    FormatPastKnowledgeController,
)
from refactored_sa_icl.usecase.schema_activation.SchemaActivationController import (
    SchemaActivationController,
)
from refactored_sa_icl.usecase.solver.SolveController import SolveController


@dataclass
class SolverResult:
    """Structured output from a solver invocation."""

    final_answer: Optional[Any]
    response_content: Any
    correct: Optional[bool]
    baseline: bool
    extra_log: Dict[str, Any]


class Solver(Protocol):
    """Callable protocol for solver implementations."""

    def __call__(
        self,
        problem: Problem,
        knowledge_base: KnowledgeBase,
        settings: ExperimentSettings,
        solver_model: Any,
        schema_activator_model: Any,
        knowledges: Optional[List[Any]],
        similarity: Optional[List[float]],
    ) -> SolverResult: ...


def _schema_aware_solver(
    problem: Problem,
    knowledge_base: KnowledgeBase,
    settings: ExperimentSettings,
    solver_model: Any,
    schema_activator_model: Any,
    knowledges: Optional[List[Any]],
    similarity: Optional[List[float]],
) -> SolverResult:
    """
    Main solver implementation mirroring the legacy _handle_single_problem
    behaviour, but driven by ExperimentSettings and explicit dependencies.
    """
    log: Dict[str, Any] = {
        "problem": str(problem.to_prompt(including_answer=False)),
        "problem_id": problem.id,
        "ground_truth": problem.get_ground_truth(),
    }

    solver_cfg = settings.solver

    # Baseline path: ignore knowledge/schema and just call the baseline use case.
    if solver_cfg.solver_type == "BaselineSolver":
        response_content, final_answer = SolveController(
            problem,
            None,
            "BaselineSolver",
            solver_model,
            None,
            None,
            settings,
        )
        correct = problem.evalute(final_answer)
        log.update(
            {
                "baseline": True,
                "final_answer": final_answer,
                "solver_response": response_content,
                "correct": correct,
            }
        )
        return SolverResult(
            final_answer=final_answer,
            response_content=response_content,
            correct=correct,
            baseline=True,
            extra_log=log,
        )

    # Non-baseline path – requires retrieved knowledge.
    if not knowledges:
        # honour fallback_mode: if enabled, drop back to baseline solver
        if solver_cfg.fallback_mode:
            response_content, final_answer = SolveController(
                problem,
                None,
                "BaselineSolver",
                solver_model,
                None,
                None,
                settings,
            )
            correct = problem.evalute(final_answer)
            log.update(
                {
                    "baseline": True,
                    "final_answer": final_answer,
                    "solver_response": response_content,
                    "correct": correct,
                }
            )
            return SolverResult(
                final_answer=final_answer,
                response_content=response_content,
                correct=correct,
                baseline=True,
                extra_log=log,
            )

        log["baseline"] = False
        log["correct"] = None
        return SolverResult(
            final_answer=None,
            response_content=None,
            correct=None,
            baseline=False,
            extra_log=log,
        )

    formatted_knowledges : list = FormatPastKnowledgeController(
        knowledges=knowledges,
        similarity=similarity,
        past_knowledge_mode=solver_cfg.past_knowledge_mode,
        settings=settings,
    )

    log["baseline"] = False
    log["relevant_knowledges"] = knowledges
    log["formatted_knowledges"] = formatted_knowledges
    log["similarity"] = similarity

    schema_for_solver: Optional[str] = None
    if solver_cfg.solver_type not in ["OneShotSolver", "CoTSolver"]:
        response_content, schema_for_solver = SchemaActivationController(
            problem,
            formatted_knowledges,
            schema_activator_type=solver_cfg.schema_activator_type,
            schema_activator_model=schema_activator_model,
            settings=settings,
        )
        log["activated_schema"] = schema_for_solver
        log["original_sensory_memory"] = getattr(
            problem, "mental_representation", None
        )
        log["schema_activation_response"] = response_content

    response_content, final_answer, messages = SolveController(
        problem,
        schema_for_solver,
        solver_cfg.solver_type,
        solver_model,
        knowledges,
        formatted_knowledges,
        settings,
    )

    correct = problem.evalute(final_answer)
    log["final_answer"] = final_answer
    log["solver_response"] = response_content
    log["correct"] = correct
    log["messages"] = messages

    return SolverResult(
        final_answer=final_answer,
        response_content=response_content,
        correct=correct,
        baseline=False,
        extra_log=log,
    )


SOLVER_REGISTRY: Dict[str, Solver] = {
    # For now all solver types are funnelled through the same implementation,
    # which internally switches based on settings.solver.solver_type. The
    # registry still allows specialised variants to be registered later.
    "default": _schema_aware_solver,
}


def get_solver(name: str = "default") -> Solver:
    """
    Fetch a registered solver by name.

    The default solver mirrors the behaviour of the legacy validation
    pipeline while allowing future extension through this registry API.
    """
    try:
        return SOLVER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown solver '{name}'. Valid options: {list(SOLVER_REGISTRY.keys())}"
        ) from exc
