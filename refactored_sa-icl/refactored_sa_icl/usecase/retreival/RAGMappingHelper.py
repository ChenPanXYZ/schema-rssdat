import numpy as np
import cohere
import json
import time
import os
from src.entity.KnowledgeBase import KnowledgeBase
from src.entity.problems.Problem import Problem
from typing import Tuple, List

try:
    from credentials import COHERE_API_KEY
except ImportError:
    COHERE_API_KEY = None


def GetRelevantKnowledgeByRerankUseCase(problem: Problem, knowledge_base: KnowledgeBase, vector_rank: int, mapping_path: str = "mappings/rerank_mapping_physics_mental.json") -> Tuple[np.array, np.array]:
    """
    Retrieves the pre-ranked knowledge item and its score from a mapping file.
    If the item is not in the mapping, it calls the Cohere API to generate it and updates the file.
    """
    time.sleep(10)
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        mapping = {}

    # Check if the result for this problem is already in our mapping
    if problem.id not in mapping:
        print(f"Problem representation not found in '{mapping_path}'. Calling API to generate and save it.")
        GenerateAndSaveRerankMappingUseCase(
            problem=problem,
            knowledge_base=knowledge_base,
            output_path=mapping_path,
            top_n=10  # Ensure we request enough items to satisfy the rank
        )
        # Reload the mapping after it has been updated
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

    # Create a lookup dictionary for fast access to knowledge items by their ID
    knowledge_lookup = {k.id: k for k in knowledge_base.knowledges}

    # Get the ranked list for the current problem's representation
    ranked_results = mapping.get(problem.id)

    if not ranked_results or len(ranked_results) < vector_rank:
        return None, None

    # Get the specific item by its rank
    target_item = ranked_results[vector_rank - 1]
    
    knowledge_id = target_item.get("id")
    relevance_score = target_item.get("relevance_score")

    # Find the full knowledge object using the ID
    relevant_knowledge = knowledge_lookup.get(knowledge_id)

    if not relevant_knowledge:
        return None, None

    return [relevant_knowledge], [relevance_score]


def GenerateAndSaveRerankMappingUseCase(problem: Problem, knowledge_base: KnowledgeBase, output_path: str, top_n: int = 10):
    """
    Generates a mapping from each problem to the top_n relevant knowledge items and saves it to a file.
    The mapping includes knowledge ID, mental representation, and relevance score.
    """
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY not found in credentials.py")

    co = cohere.Client(COHERE_API_KEY)
    documents = [k.mental_representation for k in knowledge_base.knowledges]
    knowledge_items = knowledge_base.knowledges

    try:
        with open(output_path, 'r') as f:
            mapping = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        mapping = {}

    query = problem.mental_representation
    rerank_response = co.rerank(query=query, documents=documents, top_n=top_n, model='rerank-v3.5')

    if rerank_response.results:
        top_knowledge_info = []
        for r in rerank_response.results:
            original_item = knowledge_items[r.index]
            top_knowledge_info.append({
                "id": original_item.id,
                "mental_representation": original_item.mental_representation,
                "relevance_score": r.relevance_score
            })
        mapping[problem.id] = top_knowledge_info

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=4)

    print(f"Mapping saved to {output_path}")
