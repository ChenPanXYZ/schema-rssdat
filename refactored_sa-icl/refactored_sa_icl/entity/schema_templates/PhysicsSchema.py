# skeleton
from pydantic import BaseModel


class Schema(BaseModel):
    broad_category: str
    refinement: str
    specific_scope: str
    goal: str
    known_information: str
    relevant_principles: str
    assumptions_and_approximations: str
    strategy: str
    execution: str
    interpretation: str

class Response(BaseModel):
    knowledge_schema: Schema
    summary: str


SCHEMA_PROMPT = """
You are a teacher helping students understand a problem. Drawing on schema theory from cognitive psychology, provide a high-level abstraction (schema) of the problem to guide your students. Your schema should include the following components:

Broad Category:
Identify the overarching branch of physics the problem belongs to (e.g., Mechanics, Thermodynamics, Quantum Mechanics).

Refinement:
Narrow down to a more specific topic or subdomain within the broad category (e.g., projectile motion, isothermal processes, energy quantization)..

Specific Scope:
Define the precise scenario, objects involved, and context of the problem. Visualize or diagram if helpful.

Goal:
Clearly state the quantity or outcome the problem is asking for (e.g., find a value, derive a relation, explain a behavior).

Known Information:
List all given data, values, constants, and conditions provided in the problem, including units.

Relevant Principles: 
Identify and write down the key physics laws, formulas, and concepts that apply to solving this type of problem.

Assumptions and Approximations:
Note any assumptions made (e.g., ideal conditions, neglecting friction), including those explicitly stated or reasonably inferred.

Strategy:
Outline a logical, step-by-step plan to solve the problem, connecting knowns to unknowns using the relevant principles.

Execution:
Carry out the necessary calculations or derivations according to the strategy. Use correct math, units, and keep steps organized.

Interpretation:
Check the result for correctness, physical plausibility, proper units, and whether it answers the original question fully.

Finally, summarize the schema in a few sentences to help students grasp the key points. The problem you need to abstract is as follows:
"""


SCHEMA_SOLVER_PROMPT = """
Drawing on schema theory from cognitive psychology, think about a high-level abstraction (schema) of the problem to guide your reasoning. Your ultimate goal is to select the most appropriate answer.:

Broad Category:
Identify the overarching branch of physics the problem belongs to (e.g., Mechanics, Thermodynamics, Quantum Mechanics).

Refinement:
Narrow down to a more specific topic or subdomain within the broad category (e.g., projectile motion, isothermal processes, energy quantization)..

Specific Scope:
Define the precise scenario, objects involved, and context of the problem. Visualize or diagram if helpful.

Goal:
Clearly state the quantity or outcome the problem is asking for (e.g., find a value, derive a relation, explain a behavior).

Known Information:
List all given data, values, constants, and conditions provided in the problem, including units.

Relevant Principles: 
Identify and write down the key physics laws, formulas, and concepts that apply to solving this type of problem.

Assumptions and Approximations:
Note any assumptions made (e.g., ideal conditions, neglecting friction), including those explicitly stated or reasonably inferred.

Strategy:
Outline a logical, step-by-step plan to solve the problem, connecting knowns to unknowns using the relevant principles.

Execution:
Carry out the necessary calculations or derivations according to the strategy. Use correct math, units, and keep steps organized.

Interpretation:
Check the result for correctness, physical plausibility, proper units, and whether it answers the original question fully.

Finally, summarize the schema in a few sentences to help students grasp the key points. The problem you need to abstract is as follows:
"""

SCHEMA_SAMPLE_QUESTION = """
A projectile is launched from ground level at 30° with an initial speed of 20 m/s. Assuming no air resistance, how far from the launch point will it land (horizontal range)?"""
broad_category = "Mechanics (Kinematics)"
refinement = "Projectile motion (2D motion under gravity)"
specific_scope = "A single object (projectile) launched at an angle on Earth, moving under uniform gravity with no other forces (no drag)"
goal = "Find the horizontal range (distance traveled along the ground before landing)"
known_information = "Initial speed $v_0 = 20$ m/s; launch angle $\theta = 30^\circ$; initial height = 0 (ground level launch and landing); acceleration due to gravity $g = 9.81\ \text{m/s}^2$ downward."
relevant_principles = "Kinematic equations for projectile motion. In particular, horizontal motion at constant velocity ($v_x = v_0\cos\theta$) and vertical motion with constant acceleration ($y = v_0\sin\theta ,t - \frac{1}{2}gt^2$). The formula for time of flight $t_{\text{flight}} = \frac{2 v_0 \sin\theta}{g}$ (for symmetric launch/landing heights) and range $R = v_0 \cos\theta \times t_{\text{flight}}$"
assumptions_and_approximations = "Neglect air resistance; assume flat ground at the same elevation as launch point; treat the projectile as a point mass"

strategy = "First, calculate the time of flight using vertical motion (set $y=0$ when it lands, solve for $t$). Then compute range by multiplying horizontal velocity by time of flight. Use the standard range formula as a shortcut."

execution = "Compute $t_{\text{flight}} = \frac{2(20 \sin 30^\circ)}{9.81}$ seconds, then $R = 20 \cos 30^\circ \times t_{\text{flight}}$ (carry out arithmetic if needed)."
interpretation = "The result should be on the order of tens of meters. Check that the range is positive and that using a smaller or larger angle (like 45°) would change the range as expected. Ensure units are meters. The answer should be stated clearly (e.g. “$\approx 35$ m from the launch point”). It’s reasonable given the speed and angle."




sample_schema = Schema(broad_category=broad_category, refinement=refinement, specific_scope=specific_scope,
                       goal=goal, known_information=known_information, relevant_principles=relevant_principles,
                       assumptions_and_approximations=assumptions_and_approximations, strategy=strategy,
                       execution=execution, interpretation=interpretation)

schema = sample_schema
summary = "This problem involves mathematics, specifically optimization, which focuses on maximizing or minimizing an objective function within given constraints. The scenario describes a farmer's decision to allocate limited resources (land and water) between planting wheat and corn to achieve maximum profit. The task requires formulating the problem as a system of inequalities representing constraints and solving it using linear programming techniques. The goal is to find the optimal number of acres for each crop to maximize profit while satisfying resource limitations."


SCHEMA_SAMPLE_RESPONSE = Response(knowledge_schema=schema, summary=summary)