import numpy as np
from typing import List, Tuple, Optional
from src.entity.KnowledgeBase import KnowledgeBase
from src.entity.problems.Problem import Problem
from config import solver_model

def GetRelevantKnowledgeByCoTUseCase(
    problem: Problem,
    knowledge_base: KnowledgeBase,
    top_k: int,
    exclude_self: bool = True
) -> Tuple[Optional[List], Optional[List[float]]]:
    """
    Retrieves relevant knowledge using Chain of Thought reasoning.
    The model will analyze the problem and each knowledge item to determine relevance
    through step-by-step reasoning.
    """
    if not knowledge_base.knowledges:
        return None, None

    # Prepare the prompt for the model to analyze relevance
    prompt = f"""Given the following problem and a set of knowledge items, analyze which items are most relevant using chain of thought reasoning.
    
Problem:
{problem.content}

Knowledge Items:
"""
    
    # Add each knowledge item with its content
    for i, knowledge in enumerate(knowledge_base.knowledges):
        if exclude_self and knowledge.id == problem.id:
            continue
        prompt += f"\n{i+1}. {knowledge.content}\n"

    prompt += """
Please analyze each knowledge item's relevance to the problem using chain of thought reasoning.
For each item, explain:
1. How it relates to the problem
2. What specific aspects make it relevant or irrelevant
3. A relevance score from 0 to 1

Format your response as:
Item 1:
Reasoning: [your step-by-step analysis]
Relevance Score: [0-1]

Item 2:
[repeat for each item]

Finally, list the top {top_k} most relevant items by number.
"""

    # Get model's analysis
    model = solver_model()
    response = model.generate(prompt)
    
    # Parse the response to extract relevance scores and selected items
    selected_indices = []
    relevance_scores = []
    
    # Simple parsing of the response to extract scores and selected items
    lines = response.split('\n')
    current_item = None
    current_score = None
    
    for line in lines:
        if line.startswith('Item'):
            if current_item is not None and current_score is not None:
                selected_indices.append(current_item - 1)
                relevance_scores.append(current_score)
            current_item = int(line.split()[1].replace(':', ''))
            current_score = None
        elif 'Relevance Score:' in line:
            try:
                current_score = float(line.split(':')[1].strip())
            except:
                continue
    
    # Add the last item if exists
    if current_item is not None and current_score is not None:
        selected_indices.append(current_item - 1)
        relevance_scores.append(current_score)
    
    # Get the top k items
    if len(selected_indices) > top_k:
        selected_indices = selected_indices[:top_k]
        relevance_scores = relevance_scores[:top_k]
    
    # Get the corresponding knowledge items
    selected_knowledge = [knowledge_base.knowledges[i] for i in selected_indices]
    
    return selected_knowledge, relevance_scores
