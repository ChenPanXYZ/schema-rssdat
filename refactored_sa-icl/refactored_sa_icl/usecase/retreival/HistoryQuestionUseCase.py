import numpy as np

from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
from sklearn.metrics.pairwise import cosine_similarity
from refactored_sa_icl.entity.datasets.GPQABiology import GPQABiology



def HistoryQuestionUseCase(
                                               ) -> np.array:
    """
    Runs the retrieval process to find the top-k most similar items for a target problem based on the embeddings.
    """
    # size: 1, dim
    question_text = "Which of the following best characterizes the legal and constitutional significance of the \"Calder case\" (Calder v. British Columbia, 1973) within the trajectory of Indigenous land rights in Canada?"
    incorrect_answer_1 = "It established for the first time that Indigenous title had been completely extinguished by the Royal Proclamation of 1763 and subsequent colonial legislation, thereby necessitating a modern treaty process to re-grant rights."
    incorrect_answer_2 = "While the Supreme Court was split 3-3 on whether the specific title had been extinguished, the ruling acknowledged the existence of Aboriginal title as a legal right derived from historical occupation, independent of any statute or treaty, effectively forcing the federal government to adopt a Comprehensive Land Claims policy."
    incorrect_answer_3 = "It resulted in an immediate injunction against the Nisga'a Nation preventing them from negotiating land claims, establishing the precedent that all land claims must be settled via the International Court of Justice rather than domestic Canadian courts."
    correct_answer = "It overturned the White Paper of 1969 by declaring that the Indian Act was unconstitutional, resulting in the immediate dissolution of the Department of Indian Affairs and Northern Development (DIAND)."





    # generated_problem = Problem(
    #                 id=None,
    #                 question=question_text,
    #                 context=None,
    #                 label=1,
    #                 candidates=[incorrect_answer_1, correct_answer, incorrect_answer_2, incorrect_answer_2],
    #                 explanation=None,
    #                 reference_to=None,
    #                 reference_type="history"
    #             )
    biology_problems = GPQABiology(1)
    return [biology_problems.problems[0]], None