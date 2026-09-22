from pydantic import BaseModel


class Schema(BaseModel):
    subfield: str  # e.g., "Organic Chemistry", "Thermodynamics", "Stoichiometry"
    core_concept: str  # e.g., "Reaction Mechanisms", "Ideal Gas Law", "Periodic Trends"
    question_type: str  # e.g., "Quantitative Calculation", "Conceptual Understanding", "Reaction Prediction"
    difficulty_level: str  # e.g., "High School", "Undergraduate", "Graduate"
    prerequisite_knowledge: str  # e.g., "Algebra", "Knowledge of functional groups", "Balancing equations"


class Response(BaseModel):
    knowledge_schema: Schema


SCHEMA_PROMPT = """
You are an expert chemistry professor and curriculum designer. Your task is to analyze a given chemistry question and extract its pedagogical metadata into a structured schema.

Analyze the question to determine:
1. **Subfield**: The broad area of chemistry the question falls under (e.g., Physical Chemistry, Organic Chemistry).
2. **Core Concept**: The specific scientific principle or theory being tested.
3. **Question Type**: The nature of the task (e.g., is it a math problem, a memory recall, or a mechanism design?).
4. **Difficulty Level**: The academic level required to solve it accurately.
5. **Prerequisite Knowledge**: What prior concepts or skills must the student possess to answer this?

Output your analysis strictly according to the provided JSON schema.
"""


SCHEMA_SOLVER_PROMPT = SCHEMA_PROMPT

# Below is not needed any more, but keep them as None.
SCHEMA_SAMPLE_QUESTION = None
sample_schema = None
schema = None
summary = None
SCHEMA_SAMPLE_RESPONSE = None