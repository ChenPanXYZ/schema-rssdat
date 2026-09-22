
from refactored_sa_icl.usecase.datasets.LoadProblemsUseCase import LoadProblemsUseCase


def LoadingProblemsController(problem_name, size):
    """
    Load the knowledge bases for the given dataset names.
    """
    return LoadProblemsUseCase(problem_name, size)