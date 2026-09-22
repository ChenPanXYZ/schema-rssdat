from pydantic import BaseModel


class Schema(BaseModel):
    steps: str


class Response(BaseModel):
    knowledge_schema: Schema

SCHEMA_PROMPT = """
You are a teacher helping students understand a problem.
Drawing on schema theory from cognitive psychology, provide a high-level abstraction (schema) of the problem to guide your students. Your schema should include step by step reasoning path.
**Your Schema must not include the final answer or any candidate option to the problem.**
**Keep the schema concise. Use bullet points. Limit each step to one single sentence.**
"""


SCHEMA_SOLVER_PROMPT = SCHEMA_PROMPT


# SAMPLE QUESTION & SAMPLE OUTPUT (unchanged)
SCHEMA_SAMPLE_QUESTION = None

steps = """To solve this optimization problem, we define x as acres of wheat and y as acres of corn. 
The goal is to maximize the profit function P = 200x + 300y. 
We are bound by two main constraints: 
1. Land constraint: x + y <= 100 
2. Water constraint: x + 2y <= 120 (since corn uses twice the water of wheat). 

By identifying the vertices of the feasible region, we test the corner points: 
- (100, 0) yields $20,000 
- (0, 60) yields $18,000 
- (80, 20) yields $22,000 

The optimal solution is found at the intersection of the land and water limits, 
resulting in 80 acres of wheat and 20 acres of corn for a maximum profit of $22,000.
"""
sample_schema = Schema(
    steps=steps,
)

schema = sample_schema

SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=schema)
