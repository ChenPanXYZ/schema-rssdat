from typing import List

from refactored_sa_icl.entity.problems.Problem import Problem
from refactored_sa_icl.usecase.schema_activation.NormalSchemaActivatorUseCase import (
    NormalSchemaActivatorUseCase,
)
from refactored_sa_icl.config.settings import ExperimentSettings


def SchemaActivationController(
    problem: Problem,
    knowledges_string: List,
    schema_activator_type: str,
    schema_activator_model,
    settings: ExperimentSettings,
):
    if schema_activator_type == "NormalSchemaActivator":
        response, schema = NormalSchemaActivatorUseCase(
            problem, knowledges_string, schema_activator_model, settings
        )
    else:
        raise ValueError("The schema generator type is not supported.")

    return response, schema
