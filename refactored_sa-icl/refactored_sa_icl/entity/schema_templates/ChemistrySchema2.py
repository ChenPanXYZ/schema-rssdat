from pydantic import BaseModel


class Schema(BaseModel):
    broad_category: str
    refinement: str
    specific_scope: str
    goal: str


class Response(BaseModel):
    knowledge_schema: Schema
    # summary: str


SCHEMA_PROMPT = """
You are a teacher helping students understand a problem. Drawing on schema theory from cognitive psychology, provide a high-level abstraction (schema) of the problem to guide your students. Your schema should include the following components:

Broad Category:
Identify the overarching subject and general category to which the problem belongs.

Refinement:
Describe further details or specific aspects that narrow down the broad category.

Specific Scope:
Define the precise focus or context of the problem within the refined category.

Goal:
Clearly state the objective or intended outcome of solving the problem.
"""


SCHEMA_SOLVER_PROMPT = SCHEMA_PROMPT


# ------------------------------------------------------------
# NEW: Generalized Schema Refinement Prompt (no hard-coded fields)
# ------------------------------------------------------------
# SCHEMA_REFINEMENT_PROMPT = """
# Below is the schema-definition template used in this system:
#
# Broad Category:
# Identify the overarching subject and general category to which the problem belongs.
#
# Refinement:
# Describe further details or specific aspects that narrow down the broad category.
#
# Specific Scope:
# Define the precise focus or context of the problem within the refined category.
#
# Goal:
# Clearly state the objective or intended outcome of solving the problem..
#
# You are now refining the schema you previously generated for the NEW question, by examining the schemas and summaries from previous similar questions below.
#
# %s
#
# Your task:
# Refine the NEW schema by improving clarity, removing irrelevant details,
# and incorporating only structurally compatible patterns from prior schemas.
#
# Rules:
# 1. Use exactly the same schema component names extracted from the template above.
# 2. Only integrate details from previous schemas if the tasks share structural similarity
#    (same kind of reasoning, same type of question, same goal orientation).
# 3. For any conflicts, think carefully and choose the most appropriate or generalizable detail.
# 4. Do NOT merge incompatible schema patterns or borrow structure from unrelated tasks.
# 5. Preserve the core meaning of your original schema and refine it minimally and cleanly.
# 6. Output ONLY the final refined schema, using the template’s component names.
# """


# SAMPLE QUESTION & SAMPLE OUTPUT (unchanged)
SCHEMA_SAMPLE_QUESTION = None

broad_category = "Mathematics → Optimization → Linear Programming"
refinement = "This problem focuses on maximizing profit within a given set of constraints related to resource availability (land and water)."
specific_scope = "The task involves formulating the problem using variables to represent the number of acres for each crop and solving it by applying linear programming techniques."
goal = "Determine the optimal allocation of resources (acres of land) to maximize the farmer's profit while satisfying the constraints."
sample_schema = Schema(
    broad_category=broad_category,
    refinement=refinement,
    specific_scope=specific_scope,
    goal=goal
)

schema = sample_schema
summary = (
    "This problem involves mathematics, specifically optimization, which focuses on maximizing "
    "or minimizing an objective function within given constraints. The scenario describes a farmer's "
    "decision to allocate limited resources (land and water) between planting wheat and corn to achieve "
    "maximum profit. The task requires formulating the problem as a system of inequalities representing "
    "constraints and solving it using linear programming techniques. The goal is to find the optimal number "
    "of acres for each crop to maximize profit while satisfying resource limitations."
)

SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=schema, summary=summary)
