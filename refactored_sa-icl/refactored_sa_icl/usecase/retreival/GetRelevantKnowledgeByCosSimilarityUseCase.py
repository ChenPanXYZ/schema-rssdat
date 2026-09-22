import numpy as np

from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
from sklearn.metrics.pairwise import cosine_similarity



def GetRelevantKnowledgeByCosSimilarityUseCase(problem: Problem, knowledge_base: KnowledgeBase) -> np.array:
    """
    Runs the retrieval process to find the top-k most similar items for a target problem based on the embeddings.
    """
    start_similarity, end_similarity = 0, 1.0
    # size: 1, dim
    target = problem.embedding
    # size: N, dim
    items = knowledge_base.embeddings

    # TODO: Why dimensions still not match? target is 1, 768, items is 357, 1, 768.

    target = target.reshape(1, -1)
    items = items.reshape(items.shape[0], -1)


    assert target.shape[1] == items.shape[1], "The dimensions of the target and items do not match."

    # size: N
    similarity = cosine_similarity(target, items).flatten()
    # size: N
    sorted_indices = np.argsort(similarity)[::-1]
    filtered_indices = [i for i in sorted_indices if start_similarity <= similarity[i] <= end_similarity]
    # get those filtered by threshold
    filtered_knowledges = [knowledge_base.knowledges[i] for i in filtered_indices]
    similarity = similarity[filtered_indices].tolist()
    assert len(filtered_knowledges) == len(similarity)
    if len(filtered_knowledges) == 0:
        return None, None
    return filtered_knowledges, similarity