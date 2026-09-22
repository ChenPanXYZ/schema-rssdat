from pydantic import BaseModel


class Schema(BaseModel):
    facts: str


class Response(BaseModel):
    knowledge_schema: Schema
    # summary: str

SCHEMA_PROMPT = """
You are a teacher helping students understand a problem.
Drawing on schema theory from cognitive psychology, provide a high-level abstraction (schema) of the problem to guide your students. Your schema should include a list of the general facts in a paragraph. The facts should not mention the problem itself. One sentence per fact.
"""


SCHEMA_SOLVER_PROMPT = """
You are a teacher helping students understand a problem.
Drawing on schema theory from cognitive psychology, provide a high-level abstraction (schema) of the problem to guide your students. Your schema should include a list of the general facts in a paragraph. The facts should not mention the problem itself. One sentence per fact.
"""


# SAMPLE QUESTION & SAMPLE OUTPUT (unchanged)
SCHEMA_SAMPLE_QUESTION = None

facts = """
A set of independent variables represents controllable quantities within a system.
An objective function defines a single numerical goal to be maximized or minimized.
Each variable is assigned a specific weight or value that contributes to that total goal.
Multiple constraints establish upper or lower limits on the capacity of the system.
Variables consume these limited capacities at different, measurable rates.
The optimal solution exists at the boundary where these constraints intersect or limit further growth.
"""
sample_schema = Schema(
    facts=facts,
)

schema = sample_schema
summary = ""

SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=schema, summary=summary)
