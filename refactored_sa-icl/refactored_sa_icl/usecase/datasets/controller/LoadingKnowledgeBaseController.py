from refactored_sa_icl.usecase.datasets.LoadingKnowledgeUseCase import LoadingKnowledgeUseCase


def LoadingKnowledgeBaseController(dataset_names):
    """
    Load the knowledge bases for the given dataset names.
    """
    print(dataset_names)
    return LoadingKnowledgeUseCase(dataset_names)
