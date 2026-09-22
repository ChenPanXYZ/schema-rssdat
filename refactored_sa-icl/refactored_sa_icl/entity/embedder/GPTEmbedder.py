# from typing import Optional
# import torch
# from openai import OpenAI
# import os
# from src.entity.embedder.LMEmbedder import LMEmbedder
#
# class GPTEmbedder(LMEmbedder):
#     """
#     OpenAI-based embedder using OpenAI's embedding models
#     """
#
#     def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 32,
#                  api_key: Optional[str] = None, **kwargs):
#         """
#         Initialize the OpenAI embedder.
#
#         Args:
#             model_name: The name of the OpenAI embedding model to use.
#             batch_size: Batch size for embedding generation.
#             api_key: OpenAI API key. If None, will look for OPENAI_API_KEY in environment variables.
#             **kwargs: Additional keyword arguments.
#         """
#         super().__init__(**kwargs)
#         self.model_name = model_name
#         self.batch_size = batch_size
#
#         # Initialize OpenAI client
#         self.client = OpenAI(api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))
#
#     def encode(self, texts, **kwargs) -> torch.Tensor:
#         """
#         Generate embeddings for the given texts.
#
#         Args:
#             texts: A single string or a list of strings to embed.
#             **kwargs: Additional parameters to pass to the embedding model.
#
#         Returns:
#             numpy.ndarray: The embeddings as a numpy array.
#         """
#         if isinstance(texts, str):
#             texts = [texts]
#
#         all_embeddings = []
#
#         # Process in batches to avoid API limits
#         for i in range(0, len(texts), self.batch_size):
#             batch = texts[i:i + self.batch_size]
#
#             # Call OpenAI API to get embeddings
#             response = self.client.embeddings.create(
#                 model=self.model_name,
#                 input=batch,
#                 **kwargs
#             )
#
#             # Extract embeddings from response
#             batch_embeddings = [item.embedding for item in response.data]
#             all_embeddings.extend(batch_embeddings)
#
#         # Convert to numpy array
#         return torch.Tensor(all_embeddings)
#
#     @property
#     def dimension(self) -> int:
#         """
#         Returns the dimension of the embeddings.
#
#         Returns:
#             int: The dimension of the embeddings.
#         """
#         # Different models have different dimensions, mapping common ones:
#         dimension_map = {
#             "text-embedding-ada-002": 1536,
#             "text-embedding-3-small": 1536,
#             "text-embedding-3-large": 3072,
#         }
#
#         return dimension_map.get(self.model_name, 1536)  # Default to 1536 if model not found
#
#
