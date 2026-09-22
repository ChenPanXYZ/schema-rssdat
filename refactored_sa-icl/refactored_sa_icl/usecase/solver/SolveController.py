from typing import Optional

from refactored_sa_icl.config.settings import ExperimentSettings
from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.usecase.solver.BaselineSolverUseCase import BaselineSolverUseCase
from refactored_sa_icl.usecase.solver.ComprehensiveSchemaSolver import ComprehensiveSchemaSolver
from refactored_sa_icl.usecase.solver.OneShotSchemaSolver import OneShotSchemaSolver
from refactored_sa_icl.usecase.solver.OneShotSolver import OneShotSolver
from refactored_sa_icl.usecase.solver.ZeroShotSchemaSolver import ZeroShotSchemaSolver
from refactored_sa_icl.usecase.solver.utils import apply_self_consistency


from typing import List, Dict, Any, Optional, Callable

def SolveController(
    problem: Problem,
    schema: str,
    solver_type: str,
    solver_model,
    knowledges: List[Problem],  # List of raw Problem objects
    formatted_knowledges: List[Dict[str, Any]],  # List of formatted multimodal content blocks
    settings: Optional[ExperimentSettings] = None,
):
    # print(solver_type) # Optional: Comment out if you want less console noise

    # 1. Select the base solver function
    # Ensure all these functions are updated to accept 'formatted_knowledges' as a list
    if solver_type == "ComprehensiveSchemaSolver":
        base_solver = ComprehensiveSchemaSolver
    elif solver_type == "BaselineSolver":
        base_solver = BaselineSolverUseCase
    elif solver_type == "ZeroShotSchemaSolver":
        base_solver = ZeroShotSchemaSolver
    elif solver_type == "OneShotSolver":
        base_solver = OneShotSolver
    elif solver_type == "OneShotSchemaSolver":
        base_solver = OneShotSchemaSolver
    elif solver_type == "CoTSolver":
        from refactored_sa_icl.usecase.solver.CoTSolver import CoTSolver
        base_solver = CoTSolver
    else:
        raise ValueError(f"Unsupported solver_type: {solver_type!r}")

    # 2. If self-consistency is enabled, wrap the solver
    # We pass 'formatted_knowledges' explicitly so the runner can use the pre-formatted image/text blocks
    if settings and settings.solver.use_self_consistency:
        return apply_self_consistency(
            run_once_fn=base_solver,
            problem=problem,
            schema=schema,
            solver_model=solver_model,
            knowledges=knowledges,
            formatted_knowledges=formatted_knowledges,
            settings=settings,
        )

    # 3. Normal single-run solver call
    # The base_solver functions must now handle 'formatted_knowledges' as a list of dicts
    return base_solver(
        problem,
        schema,
        solver_model,
        knowledges,
        formatted_knowledges,
        settings,
        temperature=0.0,
        top_p=1.0,
    )
