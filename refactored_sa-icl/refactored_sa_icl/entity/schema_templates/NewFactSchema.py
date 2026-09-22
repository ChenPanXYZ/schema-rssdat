from pydantic import BaseModel


class Schema(BaseModel):
    facts: str
    broad_category: str
    refinement: str
    specific_scope: str
    goal: str


class Response(BaseModel):
    knowledge_schema: Schema
    summary: str


SCHEMA_PROMPT = """
You are a teacher helping students understand a problem. Drawing on schema theory from cognitive psychology, provide a high-level abstraction (schema) of the problem to guide your students. Your schema should include the following components:

Facts:
Identify the facts that are relevant to understanding and solving the problem. One sentence per fact.

Broad Category:
Identify the overarching subject and general category to which the problem belongs using one sentence.

Refinement:
Describe further details or specific aspects that narrow down the broad category using one sentence.

Specific Scope:
Define the precise focus or context of the problem within the refined category using one sentence.

Goal:
Clearly state the objective or intended outcome of solving the problem using one sentence.

Finally, summarize the schema in a few sentences to help students grasp the key points. 
Your schema must not mention any specific details from the problem itself.
"""


SCHEMA_SOLVER_PROMPT = """
Drawing on schema theory from cognitive psychology, think about a high-level abstraction (schema) of the problem to guide your reasoning. Your ultimate goal is to select the most appropriate answer.

Below is the template for the schema you need to fill out:

Facts:
Identify the facts that are relevant to understanding and solving the problem. One sentence per fact.

Broad Category:
Identify the overarching subject and general category to which the problem belongs using one sentence.

Refinement:
Describe further details or specific aspects that narrow down the broad category using one sentence.

Specific Scope:
Define the precise focus or context of the problem within the refined category using one sentence.

Goal:
Clearly state the objective or intended outcome of solving the problem using one sentence.

Finally, summarize the schema in a few sentences to help students grasp the key points. 
Your schema must not mention any specific details from the problem itself.

The problem you need to abstract is as follows:
"""


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
SCHEMA_SAMPLE_QUESTION = """
A farmer has 100 acres of land and wants to plant two crops: wheat and corn. Each acre of wheat yields a profit of $200, and each acre of corn yields a profit of $300. However, planting corn requires twice as much water as wheat, and the farmer has only a limited supply of water that can support a maximum of 120 acres of wheat. The farmer wants to maximize their profit while using no more than the available water. How many acres of each crop should the farmer plant?
"""
facts = """
Multiple distinct activities compete for a shared set of finite resources.
Each activity generates a fixed unit of value and consumes resources at a specific rate.
The total combined usage of resources cannot exceed the strict limits available.
"""

broad_category = "The problem belongs to the general field of mathematical optimization."

refinement = "It specifically utilizes linear programming, where the relationships between variables and limits can be expressed as linear equations or inequalities."

specific_scope = "The context is a resource allocation scenario that requires balancing trade-offs between competing limitations to find an ideal intersection."

goal = "The objective is to calculate the precise quantity of each activity to perform in order to maximize the total return while satisfying all constraints."

summary = """
In this type of problem, you are essentially a manager with a "budget" of different resources (like time, space, or money). You have several ways to spend that budget, and each way gives you a different reward but costs a different amount. Your task is to find the "sweet spot"—the perfect combination of activities—that gives you the highest possible reward without running out of any single resource.
"""

sample_schema = Schema(
    facts = facts,
    broad_category=broad_category,
    refinement=refinement,
    specific_scope=specific_scope,
    goal=goal
)

schema = sample_schema

SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=schema, summary=summary)
