# import os, pickle
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import Optional
#
# from src.embedder.STEmbedder import STEmbedder
# from src.retriever.abstractRetriever import AbstractRetriever
#
#
# class DenseRetriever(AbstractRetriever):
#     """
#     Concrete DenseRetriever class.
#     """
#     cls_type = "dense"
#     def __init__(self, mode: str, embeddings_dir: str, k, model: Optional[
#         STEmbedder] = None, INSTRUCTOR_THRESHOLD_MIN = 0, INSTRUCTOR_THRESHOLD_MAX = 1):
#         super().__init__(mode=mode, embeddings_dir=embeddings_dir,
#                          embedding_model=model, k=k, INSTRUCTOR_THRESHOLD_MIN=INSTRUCTOR_THRESHOLD_MIN, INSTRUCTOR_THRESHOLD_MAX=INSTRUCTOR_THRESHOLD_MAX)
#
#     def load_data(self) -> dict[str, np.ndarray]:
#         dest_embs = self.load_dest_embeddings()
#         return dest_embs
#
#     def retrieval_for_query(self, problems) -> tuple[float, list[str]]:
#         """
#         Compute the similarity between the problem and other problems.
#         """
#         return self.run_retrieval(problems = problems)