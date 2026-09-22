from pydantic import BaseModel, Field


class Schema(BaseModel):
    # 1. IDENTIFICATION (Jonassen):
    # Categorizes the problem type to trigger the correct 'mental set'.
    problem_taxonomy: str = Field(
        ...,
        description=(
            "The structural category of the problem. This must be independent of the "
            "specific domain."
        )
    )

    # 2. ELABORATION (Chi & VanLehn - Surface Features):
    # Isolates the 'cover story' so it can be discarded or swapped during analogical transfer.
    surface_narrative: str = Field(
        ...,
        description=(
            "The specific context, entities, and scenario of the problem (the 'Surface Features'). "
            "Include the explicit objects here. This represents "
            "the variable content that changes from case to case."
        )
    )

    # 3. DEEP STRUCTURE (Gentner - Relational Structure):
    # Captures the causal or mathematical relationships using abstract roles.
    structural_logic: str = Field(
        ...,
        description=(
            "The abstract logical or causal relationships that govern the system, stripped of specific nouns. "
            "Use abstract roles. "
        )
    )

    # 4. PLANNING (Marshall - Goals & Constraints):
    # Defines the success state and the boundaries.
    goal_constraints: str = Field(
        ...,
        description=(
            "A precise definition of the Goal State (what needs to be achieved) and the Constraints "
            "(limitations, boundaries, or forbidden states). This defines the 'search space' for the solution."
        )
    )

    # 5. EXECUTION (Polya/Newell - Heuristics):
    # Generalized strategies, NOT just step-by-step algorithms.
    heuristic_prototype: str = Field(
        ...,
        description=(
            "The general problem-solving strategy or heuristic used to derive the solution. "
        )
    )


class Response(BaseModel):
    # Metacognitive trace to ensure the LLM 'thinks' before populating the schema.
    # reasoning_trace: str = Field(
    #     ...,
    #     description="A brief explanation of how the surface features were separated from the deep structure."
    # )
    knowledge_schema: Schema


# ==========================================
# SYSTEM PROMPTS (Generalized)
# ==========================================

SCHEMA_PROMPT = """
You are an expert cognitive scientist and systems architect. Your goal is to analyze a problem and extract its 'Schema'—the abstract mental template used to solve it.

You must perform 'Semantic Segregation':
1. Identify the 'Problem Taxonomy'.
2. Strip away the 'Surface Narrative'.
3. Distill the 'Structural Logic'.
4. Define the 'Goal & Constraints'.
5. Extract the 'Heuristic Prototype'.

Your output must be GENERAL. The 'Structural Logic' and 'Heuristic' should be applicable to *any* problem of this type, not just the specific example provided.

**Your Schema must not include the final answer or any candidate option to the problem.**
**Keep the schema concise. Limit each step to one single sentence.**
"""

SCHEMA_SOLVER_PROMPT = SCHEMA_PROMPT

# ==========================================
# SAMPLE DATA (Demonstrating Generality)
# ==========================================

SCHEMA_SAMPLE_QUESTION = None

# Note: The values below are specific to the sample to show it matches,
# but the phrasing emphasizes the *structure* over the *story*.

sample_schema = Schema(
    problem_taxonomy="Constrained Optimization (Linear Programming)",

    surface_narrative="A farmer must allocate land to Wheat and Corn to make profit, given distinct water requirements for each and a total water limit.",

    structural_logic=(
        "1. Variables: Defined independently (x, y) with associated weights (profit coefficients).\n"
        "2. Constraints: Resource usage is linear (a*x + b*y <= Limit).\n"
        "3. Objective: Maximize the weighted sum of variables.\n"
        "4. Optimal State: Located at the vertex of the feasible region defined by constraints."
    ),

    goal_constraints=(
        "Goal: Maximize Total Value (Profit).\n"
        "Constraint 1 (Hard): Total Primary Resource (Land) <= 100.\n"
        "Constraint 2 (Hard): Total Secondary Resource (Water) <= Equivalent of 120 units."
    ),

    heuristic_prototype=(
        "Strategy: Graphical Analysis or Simplex Method.\n"
        "1. Define decision variables for the unknown quantities.\n"
        "2. Formulate the Objective Function and Inequality Constraints.\n"
        "3. Identify the 'Feasible Region' (the overlap of all constraints).\n"
        "4. Evaluate the Objective Function at the 'Corner Points' (vertices) of the region to find the maximum."
    )
)

schema = sample_schema

SCHEMA_SAMPLE_RESPONSE = Response(
    reasoning_trace="Separated agricultural context from the mathematical optimization logic. Identified the problem as a standard Linear Programming task solvable via corner-point evaluation.",
    knowledge_schema=schema
)
