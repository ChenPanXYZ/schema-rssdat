from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
import random
from refactored_sa_icl.usecase.retreival.GetReferenceQuestionUseCase import (
    GetReferenceQuestionUseCase,
)
from refactored_sa_icl.usecase.retreival.GetRelevantKnowledgeByCosSimilarityUseCase import (
    GetRelevantKnowledgeByCosSimilarityUseCase,
)
from refactored_sa_icl.usecase.retreival.GetRelevantKnowledgeByRerankUseCase import (
    GetRelevantKnowledgeByRerankUseCase,
)
from refactored_sa_icl.usecase.retreival.GetRelevantKnowledgeInternalUseCase import (
    GetRelevantKnowledgeInternalUseCase,
)

from refactored_sa_icl.usecase.retreival.HistoryQuestionUseCase import (
    HistoryQuestionUseCase,
)



def _filter(problem: Problem, knowledges: [Problem], similarities: [float], settings: ExperimentSettings):
    top_k = settings.dataset.top_k
    exclude = settings.retrieval.exclude # ['self', 'paraphrase', 'new_question', 'new_question_exam']
    random_from_k: bool = settings.retrieval.random_from_k
    num_shots = settings.solver.num_shots

    start_similarity = settings.retrieval.start_similarity
    end_similarity = settings.retrieval.end_similarity

    # should get rid of exclude first then do top_k and random_from_k.
    original_synthetic_mappings = settings.dataset.original_synthetic_mappings
    removed_indices = []
    for idx, knowledge in enumerate(knowledges):
        problem_id = problem.id
        knowledge_id = knowledge.id
        for e in exclude:
            assert problem_id in original_synthetic_mappings, f"Problem ID {problem_id} not found in original_synthetic_mappings."
            if original_synthetic_mappings[problem_id].get(e, None) == knowledge_id:
                removed_indices.append(idx)
                break

    # remove these indices from both knowledges and similarities
    knowledges = [k for i, k in enumerate(knowledges) if i not in removed_indices]
    similarities = [s for i, s in enumerate(similarities) if i not in removed_indices]

    # now filter by similarity range
    filtered_knowledges = []
    filtered_similarities = []
    for k, s in zip(knowledges, similarities):
        if start_similarity <= s <= end_similarity:
            filtered_knowledges.append(k)
            filtered_similarities.append(s)
    knowledges = filtered_knowledges
    similarities = filtered_similarities

    # now get top_k
    assert top_k is not None and num_shots is not None, "top_k and num_shots must be set in settings, or given a default value."
    if num_shots > top_k:
        top_k = num_shots

    knowledges = knowledges[:top_k]
    similarities = similarities[:top_k]

    # now, get num_shots from random_from_k
    assert random_from_k is not None, "random_from_k must be set in settings, or given a default value."

    if random_from_k and len(knowledges) > num_shots:

        combined = list(zip(knowledges, similarities))
        selected = random.sample(combined, num_shots)
        knowledges, similarities = zip(*selected)
        knowledges = list(knowledges)
        similarities = list(similarities)

    return knowledges, similarities


def GetRelevantKnowledgesController(
    problem: Problem,
    knowledge_base: KnowledgeBase,
    settings: ExperimentSettings | None = None,
    solver_model = None
):
    approach = settings.retrieval.approach
    mapping_path = settings.retrieval.mapping_path

    if approach == "cos_similarity":
        # load start_similarity, end_similarity
        knowledges, similarity = GetRelevantKnowledgeByCosSimilarityUseCase(problem, knowledge_base)

    elif approach == "rag_rerank":
        knowledges, similarity = GetRelevantKnowledgeByRerankUseCase(
            problem,
            knowledge_base,
            mapping_path
        )

    elif approach in ["paraphrase", "new_question", "new_question_exam"]:
        knowledges, similarity = GetReferenceQuestionUseCase(problem, knowledge_base, approach)

        return knowledges, similarity # TODO: Similarity will be None here as we don't care about similarity an

    elif approach == "internal":
        knowledges, similarity = GetRelevantKnowledgeInternalUseCase(problem, solver_model)
        return knowledges, similarity
    elif approach == "no_knowledge":
        return None, None
    elif approach == "cot":
        raise Exception("Chain of Thought (CoT) retrieval approach not implemented")
    elif approach == "history":
        knowledges, similarity = HistoryQuestionUseCase()
        return knowledges, similarity
    else:
        raise Exception("Retrival Approach not supported")

    assert len(knowledges) == len(similarity), "Length of knowledges and similarity must be the same"

    knowledges, similarity = _filter(problem, knowledges, similarity, settings)
    return knowledges, similarity
