import os

from src.entity.KnowledgeBase import KnowledgeBase
from src.entity.embedder.STEmbedder import STEmbedder
from src.usecase.datasets.CreateEmbeddingUseCase import CreateKnowledgeBaseEmbeddingUseCase
from root import root
from src.usecase.datasets.LoadEmbeddingUseCase import LoadEmbeddingUseCase


def CreateKnowledgeBaseEmbeddingController(embedder: STEmbedder, knowledge_base: KnowledgeBase, memory_type, memory_model):
    """
    Create embeddings for the given knowledge base.
    """
    embedding_path = f'{root}/embeddings/{memory_type}_by_{memory_model}_{embedder.model_name.replace("/", "@")}'
    # check if the embedding directory exists
    if os.path.exists(embedding_path):
        return LoadEmbeddingUseCase(knowledge_base, embedding_path)
    else:
        # create the path.
        os.makedirs(embedding_path, exist_ok=True)
        return CreateKnowledgeBaseEmbeddingUseCase(embedder, knowledge_base, embedding_path)