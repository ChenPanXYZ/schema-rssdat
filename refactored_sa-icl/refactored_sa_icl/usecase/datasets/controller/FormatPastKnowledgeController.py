from __future__ import annotations

from typing import List

from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.usecase.datasets.FormatPastKnowledgeWithQuestionAndAnswerUseCase import (
    FormatPastKnowledgeWithQuestionAndAnswerUseCase,
)
from refactored_sa_icl.config.settings import ExperimentSettings


def FormatPastKnowledgeController(
    knowledges: List[Problem],
    similarity: List[float],
    past_knowledge_mode: str,
    settings: ExperimentSettings,
) -> List:
    """
    Format past knowledge according to the configured mode.

    This version is decoupled from global Settings and instead receives the
    immutable ExperimentSettings instance explicitly.
    """

    if past_knowledge_mode == "question+answer":
        num_shots = settings.solver.num_shots
        return FormatPastKnowledgeWithQuestionAndAnswerUseCase(
            settings, knowledges, num_shots=num_shots
        )

    raise ValueError(f"Unsupported past_knowledge_mode: {past_knowledge_mode!r}")
