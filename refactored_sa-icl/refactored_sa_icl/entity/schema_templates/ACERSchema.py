# skeleton
from pydantic import BaseModel

# https://arxiv.org/pdf/1611.02262
class Schema(BaseModel):
    activation: str
    construction: str
    execution: str
    reflection: str

class Response(BaseModel):
    knowledge_schema: Schema
    summary: str


SCHEMA_PROMPT = """
You are a teacher helping students understand a problem. Drawing on schema theory from cognitive psychology, provide a high-level abstraction (schema) of the problem to guide your students. Your schema should include the following components:

Activation: 
Determine the proper mathematical tool.

Construction:
Make a mathematical model.

Execution:
Perform mathematical steps.

Reflection:
Check the final solution.

Finally, summarize the schema in a few sentences to help students grasp the key points. The problem you need to abstract is as follows:
"""

SCHEMA_SOLVER_PROMPT = """
Drawing on schema theory from cognitive psychology, think about a high-level abstraction (schema) of the problem to guide your reasoning. Your ultimate goal is to select the most appropriate answer.:

Activation: 
Determine the proper mathematical tool.

Construction:
Make a mathematical model.

Execution:
Perform mathematical steps.

Reflection:
Check the final solution.

Finally, summarize the schema in a few sentences to help students grasp the key points. The problem you need to abstract is as follows:
"""

SCHEMA_SAMPLE_QUESTION = """A farmer has 100 acres of land and wants to plant two crops: wheat and corn. Each acre of wheat yields a profit of $200, and each acre of corn yields a profit of $300. However, planting corn requires twice as much water as wheat, and the farmer has only a limited supply of water that can support a maximum of 120 acres of wheat. The farmer wants to maximize their profit while using no more than the available water. How many acres of each crop should the farmer plant?"""

activation = "We recognize this as an optimization problem because we need to decide on the number of acres for each crop (wheat and corn) in order to maximize profit, subject to constraints on land and water usage. This naturally leads us to use linear programming as our mathematical tool."
construction = "We define decision variables x (acres of wheat) and y (acres of corn). Our objective function is to maximize profit: Maximize 200x + 300y. We have two main constraints: (1) Limited acreage: x + y ≤ 100, and (2) Limited water: x + 2y ≤ 120. We also have non-negativity: x ≥ 0, y ≥ 0."
execution = "To solve, we find the corner points of the feasible region. Checking the constraint intersections, we solve x + y = 100 and x + 2y = 120 simultaneously. This gives x=80, y=20. At this point, the profit is 200(80) + 300(20) = 22,000. Checking other corners shows that this is the maximum possible profit. Therefore, the farmer should plant 80 acres of wheat and 20 acres of corn."
reflection = "We verify the solution by confirming all constraints are satisfied: 80 + 20 = 100 acres of land used, and 80 + 2×20 = 120 units of water used, which matches the water limit. No constraints are violated, and the profit is properly calculated."
summary = "By formulating a linear program, we maximize profit subject to land and water constraints. The optimal solution is 80 acres of wheat and 20 acres of corn for a profit of $22,000."



sample_schema = Schema(
    activation=activation,
    construction=construction,
    execution=execution,
    reflection=reflection
)
schema = sample_schema



SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=schema, summary=summary)