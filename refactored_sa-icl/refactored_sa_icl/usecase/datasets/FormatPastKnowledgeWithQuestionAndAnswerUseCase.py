import base64
from typing import List, Dict, Any


def _merge_adjacent_text_blocks(content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper to merge consecutive text blocks in the content list."""
    if not content_list:
        return []

    merged = []
    current_text = ""

    for item in content_list:
        if item["type"] == "text":
            # Accumulate text
            current_text += item["text"]
        else:
            # If we hit a non-text block (like image), push the accumulated text first
            if current_text:
                merged.append({"type": "text", "text": current_text})
                current_text = ""
            merged.append(item)

    # Push any remaining text
    if current_text:
        merged.append({"type": "text", "text": current_text})

    return merged

def FormatPastKnowledgeWithQuestionAndAnswerUseCase(
        settings,
        knowledges: List[Any],
        num_shots: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Format a list of past knowledge items as a list of content blocks (text + images).
    If `num_shots` is provided, only the first `num_shots` items will be included.
    """
    formatted_content = []

    # Slice the knowledge list if num_shots is provided
    past_knowledges = knowledges if num_shots is None else knowledges[:num_shots]


    for i, past_knowledge in enumerate(past_knowledges):
        # 1. Get the base content (Question, Candidates, Image, Answer)
        # We use the existing to_prompt method which handles the image logic.
        item_content = past_knowledge.to_prompt(including_answer=(True and settings.solver.include_answer_in_example))

        # 2. Append the Mental Representation (Schema/Summary)
        # The original string function included this, but to_prompt does not.
        # We add it as a new text block.
        mental_rep_text = (
            f"{past_knowledge.mental_representation}"
        )

        item_content.append({
            "type": "text",
            "text": mental_rep_text
        })

        # 3. Add a separator between shots (but not after the last one)
        # This helps the LLM distinguish between different examples.
        if i < len(past_knowledges) - 1:
            item_content.append({
                "type": "text",
                "text": "\n\n---\n\n"
            })

        # 4. Extend the main formatted_content list
        formatted_content.extend(item_content)

    return _merge_adjacent_text_blocks(formatted_content)
