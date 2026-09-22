from src.entity.KnowledgeBase import KnowledgeBase
from src.entity.embedder.STEmbedder import STEmbedder

def CreateKnowledgeBaseEmbeddingUseCase(embedder: STEmbedder, knowledge_base: KnowledgeBase, embedding_path: str):
    """
    Create embeddings for the given knowledge base.
    """
    knowledges = knowledge_base.knowledges
    return embedder.create_embeddings(problems=knowledges, mode="st", output_dir=embedding_path)