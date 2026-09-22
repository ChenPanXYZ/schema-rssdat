# import abc, os, pickle
# import torch
# from tqdm import tqdm
# import tqdm
#
# from refactored_sa_icl.entity.problems.Problem import Problem
#
# SPLIT_TYPE = {"sentence", "section"}
#
#
# class LMEmbedder(abc.ABC):
#     """
#     Abstract base class for a Embedder.
#     """
#
#     def __init__(self, model_name: str):
#         self.model_name = model_name
#
#     @abc.abstractmethod
#     def encode(self, text: str | list[str]) -> torch.Tensor:
#         """
#         Encode the given text into embeddings.
#
#         text (str or list[str]): Text or list of text segments to be encoded.
#         """
#         pass
#
#     def create_embeddings(self, problems: list[Problem], mode: str, output_dir: str):
#         """
#         Create embeddings for the given dataset.
#
#         mode: str, "question + context" or "question" or "question + answer"
#         """
#         # show progress bar
#         for i, problem in enumerate(tqdm.tqdm(problems)):
#             # TODO: add more modes
#             text = str(problem)
#             with torch.no_grad():
#                 embeddings = self.encode(text)
#             id = problem.id
#             with open(os.path.join(output_dir, f"{id}.pkl"), "wb") as f:
#                 pickle.dump(embeddings, f)
#         return embeddings
