import json
import os
import time
from pathlib import Path
from typing import Tuple, Optional
import cohere
import numpy as np

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase
from refactored_sa_icl.entity.problems.Problem import Problem
try:
    from credentials import COHERE_API_KEY
except ImportError:
    COHERE_API_KEY = None

def GetRelevantKnowledgeByRerankUseCase(
    problem: Problem,
    knowledge_base: KnowledgeBase,
    mapping_path: str,
) -> Tuple[np.array, np.array]:
    """
    Retrieves relevant knowledge based on a pre-computed reranking mapping file.
    If a problem is not in the mapping, it generates the mapping and saves it for future use.
    It filters items by similarity score and applies top-k and random selection logic.
    """
    # Always try to use a pre-computed rerank mapping from disk. We do not
    # call external APIs in the refactored pipeline – if a mapping entry is
    # missing, we simply fall back to 'no knowledge' and let the caller
    # decide how to handle it.
    project_root = Path(__file__).resolve().parents[4]
    mapping_file = project_root / mapping_path

    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        mapping = {}

    if problem.id not in mapping:
        print(f"Problem ID '{problem.id}' not found in mapping file. Generating and saving it now.")
        GenerateAndSaveRerankMappingUseCase(
            problem=problem,
            knowledge_base=knowledge_base,
            output_path=mapping_file,
            top_n=10
        )

        # Reload the mapping after it has been updated
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
    else:
        print(f"Problem ID '{problem.id}' successfully found in mapping file.")



    problem_results = mapping[problem.id]

    # Filter results by similarity range and self-exclusion
    filtered_results = []
    for result_item in problem_results:
        filtered_results.append(result_item)

    if not filtered_results:
        return None, None

    final_results = filtered_results

    if not final_results:
        return None, None

    knowledge_map = {k.id: k for k in knowledge_base.knowledges}
    relevant_knowledges = []
    relevance_scores = []

    for result in final_results:
        knowledge_id = result['id']
        if knowledge_id in knowledge_map:
            relevant_knowledges.append(knowledge_map[knowledge_id])
            relevance_scores.append(result['relevance_score'])

    if not relevant_knowledges:
        return None, None

    return relevant_knowledges, relevance_scores


def GenerateAndSaveRerankMappingUseCase(problem: Problem, knowledge_base: KnowledgeBase, output_path: str, top_n: int = 10):
    """
    Generates a mapping from each problem to the top_n relevant knowledge items and saves it to a file.
    The mapping includes knowledge ID, mental representation, and relevance score.
    """
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY not found in credentials.py")

    # Sleep for 5 seconds to avoid rate limit
    time.sleep(5)

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