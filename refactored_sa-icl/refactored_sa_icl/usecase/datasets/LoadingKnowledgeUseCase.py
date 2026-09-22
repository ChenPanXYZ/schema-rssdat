from refactored_sa_icl.entity.datasets import load_dataset_classes
from refactored_sa_icl.entity.KnowledgeBase import KnowledgeBase


def LoadingKnowledgeUseCase(dataset_names: [str]):
    registry = load_dataset_classes()
    knowledge_bases = []

    for dataset_name in dataset_names:
        if dataset_name not in registry:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        dataset = registry[dataset_name](size=100000000)
        knowledge_bases.extend(dataset.problems)

    return KnowledgeBase(dataset_names, knowledge_bases)
