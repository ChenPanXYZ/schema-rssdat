# import torch
# from sentence_transformers import SentenceTransformer
#
# from refactored_sa_icl.entity.embedder.LMEmbedder import LMEmbedder
#
#
# class STEmbedder(LMEmbedder):
#     """
#     Embedder using sentence transformer.
#     """
#
#     def __init__(self, model_name: str, split_type: str = "section") -> None:
#         # if no model name is provided, use the default one.
#         self.model_name = model_name
#         super().__init__(model_name=model_name)
#         self.model = SentenceTransformer(model_name, trust_remote_code=True)
#
#     def encode(self, text: str) -> torch.Tensor:
#         with torch.no_grad():
#             return self.model.encode([text] if isinstance(text, str) else text)
#
#
