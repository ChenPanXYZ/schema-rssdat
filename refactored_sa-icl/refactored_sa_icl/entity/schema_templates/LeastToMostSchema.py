from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. DATA MODELS (Renamed for Least-to-Most)
# ---------------------------------------------------------

class DecompositionSchema(BaseModel):
    sub_questions: str = Field(
        description="A numbered list of sequential sub-questions needed to solve the problem."
    )

class Response(BaseModel):
    knowledge_schema: DecompositionSchema
    # Summary removed as requested

# ---------------------------------------------------------
# 2. PROMPTS (The Key "Least-to-Most" Logic)
# ---------------------------------------------------------

SCHEMA_PROMPT = """
You are an expert problem solver using the 'Least-to-Most' prompting strategy.
Your goal is to break down the user's complex problem into a list of simple, sequential sub-questions.

1. Read the problem carefully.
2. Identify the logical steps required to solve it.
3. Output a numbered list of sub-questions in the 'sub_questions' field.
   - The sub-questions MUST be specific to the problem (mention specific numbers or entities).
   - Do NOT solve the sub-questions yet.
   - Ensure the last sub-question asks for the final answer.
"""

SCHEMA_SOLVER_PROMPT = SCHEMA_PROMPT  # Can be the same for consistency

# ---------------------------------------------------------
# 3. FEW-SHOT EXAMPLES (Updated for Decomposition)
# ---------------------------------------------------------

# The sample question remains the same
SCHEMA_SAMPLE_QUESTION = None

# The output is now a specific plan, not abstract facts
ltm_decomposition_steps = """
1. Define the variables for the number of acres of wheat (x) and corn (y).
2. Write the objective function for total profit (P = 200x + 300y).
3. Write the constraint inequality for total land area (x + y <= 100).
4. Write the constraint for water usage, converting corn's water needs into wheat-equivalent units (x + 2y <= 120).
5. Identify the non-negativity constraints (x >= 0, y >= 0).
6. Find the vertices of the feasible region defined by these inequalities.
7. Evaluate the profit function P at each vertex to find the maximum profit.
"""

sample_schema = DecompositionSchema(
    sub_questions=ltm_decomposition_steps,
)

SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=sample_schema)