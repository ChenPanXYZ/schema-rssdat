# import os, pickle, json, abc
# import string
# import time
# import random
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Model is a simple class that handles the API requests to different end services.
# import sqlite3
#
# from src.embedder.LMEmbedder import LMEmbedder
# from src.problems.Problem import Problem
#
#
# class AbstractRetriever(abc.ABC):
#     """
#     Abstract base dense retriever class.
#     """
#     def __init__(self, mode:str, embeddings_dir: str, embedding_model: LMEmbedder, k: int = 5, INSTRUCTOR_THRESHOLD_MIN = 0, INSTRUCTOR_THRESHOLD_MAX = 1):
#         """
#         mode: str, "question + context" or "question" or "question + answer"
#         """
#         self.embedding_model = embedding_model
#         self.embeddings_dir = embeddings_dir
#         self.k = k
#         self.mode = mode
#         self.INSTRUCTOR_THRESHOLD_MIN = INSTRUCTOR_THRESHOLD_MIN
#         self.INSTRUCTOR_THRESHOLD_MAX = INSTRUCTOR_THRESHOLD_MAX
#
#
#     def sort_past_knowledge(self, problem: Problem) -> np.array:
#         """
#         Runs the retrieval process to find the top-k most similar items for a target problem based on the embeddings.
#         """
#         # size: 1, dim
#         target = self.embedding_model.encode(str(problem))
#         # size: N, dim
#         items = self.load_item_embedding()
#
#         assert target.shape[1] == items.shape[1], "The dimensions of the target and items do not match."
#
#         # size: N
#         similarity = cosine_similarity(target, items).flatten()
#         # size: N
#         sorted_indices = np.argsort(similarity)[::-1]
#
#         from exp_config import EXCLUDE_SELF
#         my_index = Problem.get_problem_index(problem)
#         if EXCLUDE_SELF:
#             filtered_indices = sorted_indices[(similarity >= self.INSTRUCTOR_THRESHOLD_MIN) & (similarity <= self.INSTRUCTOR_THRESHOLD_MAX) & (sorted_indices != my_index)]
#         else:
#             filtered_indices = sorted_indices[(similarity >= self.INSTRUCTOR_THRESHOLD_MIN) & (similarity <= self.INSTRUCTOR_THRESHOLD_MAX)]
#         # get those filtered by threshold
#         return filtered_indices, similarity