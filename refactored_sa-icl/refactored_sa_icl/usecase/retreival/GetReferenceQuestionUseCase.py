import numpy as np

from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
from sklearn.metrics.pairwise import cosine_similarity



def GetReferenceQuestionUseCase(problem: Problem, knowledge_base: KnowledgeBase, reference_type
                                               ) -> np.array:
    """
    Runs the retrieval process to find the top-k most similar items for a target problem based on the embeddings.
    """
    # size: 1, dim

    for knowledge in knowledge_base.knowledges:
        if knowledge.reference_to != problem.id or knowledge.reference_type != reference_type:
            continue
        # target = knowledge.embedding
        # item = problem.embedding
        # similarity = cosine_similarity(target.reshape(1, -1), item.reshape(1, -1)).flatten()
        return [knowledge], None # TODO we don't care about similarity here
