from refactored_sa_icl.entity.datasets import load_dataset_classes


def LoadProblemsUseCase(dataset_name: str, size: int):
    """
    Load problems for a specific dataset, without hardcoding dataset names.
    """
    dataset_registry = load_dataset_classes()

    if dataset_name not in dataset_registry:
        raise ValueError(f"Dataset {dataset_name} is not supported. "
                         f"Available: {list(dataset_registry.keys())}")

    dataset_class = dataset_registry[dataset_name]
    dataset = dataset_class(size=size)

    return dataset.problems
