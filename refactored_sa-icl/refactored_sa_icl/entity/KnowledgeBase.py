from typing import List

import numpy as np

from refactored_sa_icl.entity.problems.Problem import Problem


class KnowledgeBase:
    name: str
    knowledges: List[Problem]

    def __init__(self, dataset_names: List[str], knowledges: List[Problem]) -> None:
        self.name = "_".join(dataset_names)
        self.knowledges = knowledges
        self.embeddings: np.ndarray | List[np.ndarray] = []

    def add_embeddings(self) -> None:
        """Materialise a dense matrix of embeddings from the problems."""
        return
        self.embeddings = [knowledge.embedding for knowledge in self.knowledges]
        self.embeddings = np.array(self.embeddings)

    def get_problem_index(self, problem: Problem) -> int:
        """Return the index of ``problem`` in the knowledge base, or -1."""
        for existing in self.knowledges:
            if str(existing) == str(problem):
                return self.knowledges.index(existing)
        return -1
