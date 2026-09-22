from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.usecase.retreival.GetRelevantKnowledgesController import (
    GetRelevantKnowledgesController,
)


@dataclass
class RetrievalResult:
    """Result of a retrieval call."""

    knowledges: Optional[List[Any]]
    similarity: Optional[List[float]]


class KnowledgeRetriever(Protocol):
    """Callable protocol for retrieval implementations."""

    def __call__(
        self,
        problem: Problem,
        knowledge_base: KnowledgeBase,
        settings: ExperimentSettings,
    ) -> RetrievalResult: ...


def _default_retriever(
    problem: Problem,
    knowledge_base: KnowledgeBase,
    settings: ExperimentSettings,
    solver_model: Any
) -> RetrievalResult:
    """
    Single entrypoint that delegates to the existing
    GetRelevantKnowledgesController using values from ExperimentSettings.
    """
    r = settings.retrieval

    knowledges, similarity = GetRelevantKnowledgesController(
        problem=problem,
        knowledge_base=knowledge_base,
        settings=settings,
        solver_model=solver_model
    )

    # GetRelevantKnowledgesController may return (None, None) depending on
    # the configured approach.
    return RetrievalResult(
        knowledges=knowledges,
        similarity=similarity,
    )


RETRIEVAL_REGISTRY: Dict[str, KnowledgeRetriever] = {
    "default": _default_retriever,
}


def get_retriever(name: str = "default") -> KnowledgeRetriever:
    """
    Fetch a registered retriever by name.

    Currently only a single default implementation is registered, but this
    registry-based API allows additional specialised retrievers to be added
    without changing the pipeline.
    """
    try:
        return RETRIEVAL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown retriever '{name}'. Valid options: {list(RETRIEVAL_REGISTRY.keys())}"
        ) from exc


