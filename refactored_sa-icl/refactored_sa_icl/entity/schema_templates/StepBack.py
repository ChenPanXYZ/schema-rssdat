from pydantic import BaseModel, Field

class Schema(BaseModel):
    step_back_question: str = Field(..., description="The high-level abstract question derived from the specific problem.")
    principle: str = Field(..., description="The general principle, law, or fact that answers the step-back question.")
    steps: str = Field(..., description="The reasoning path applying the principle to the specific problem.")

class Response(BaseModel):
    knowledge_schema: Schema

SCHEMA_PROMPT = """
You are an expert teacher helping students understand a problem using **Step-Back Prompting**.
Instead of solving the specific details immediately, you must first abstract the problem to a higher level.

Please provide the following:
1. **Step-Back Question**: A generic, high-level question that addresses the underlying concept, theory, or physics law behind the specific problem. This question should be abstract enough to apply to a class of similar problems.
2. **Principle**: The fundamental principle, formula, or fact required to answer that step-back question.
3. **Reasoning Steps**: A concise, step-by-step outline of how to apply that principle to solve the specific problem.

**Your Schema must not include the final answer or any candidate option.**
**Keep the steps concise. Use bullet points.**
"""

SCHEMA_SOLVER_PROMPT = SCHEMA_PROMPT

# SAMPLE DATA FOR TESTING
SCHEMA_SAMPLE_QUESTION = None

# Example based on the Optimization Problem (Wheat/Corn)
step_back_q = "How do you find the maximum value of a linear function subject to linear constraints?"

principle = "The Fundamental Theorem of Linear Programming states that if an optimal solution exists, it will occur at one of the vertices (corner points) of the feasible region defined by the constraints."

steps = """- Define variables x and y for the quantities of each crop.
- Formulate the objective function P (Profit) in terms of x and y.
- Write down the linear inequalities for all constraints (Land and Water).
- Graph or calculate the intersection points to identify the vertices of the feasible region.
- Evaluate the objective function P at each vertex coordinate.
- Select the vertex that yields the highest value for P."""

sample_schema = Schema(
    step_back_question=step_back_q,
    principle=principle,
    steps=steps,
)

schema = sample_schema

SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=schema)
