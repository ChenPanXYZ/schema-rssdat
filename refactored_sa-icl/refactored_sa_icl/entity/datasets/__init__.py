from refactored_sa_icl.entity.datasets.Dataset import Dataset
import pkgutil
import importlib
import inspect


def load_dataset_classes():
    """
    Automatically discover all dataset classes that inherit from Dataset.
    """
    import refactored_sa_icl.entity.datasets as datasets_pkg

    dataset_classes = {}

    for _, module_name, _ in pkgutil.iter_modules(datasets_pkg.__path__):
        module = importlib.import_module(f"{datasets_pkg.__name__}.{module_name}")

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Dataset) and obj is not Dataset:
                dataset_classes[name] = obj

    return dataset_classes

def load_problem_ids(dataset_name: str, size: int = 1_000_000) -> list:
    """
    Load problem IDs for any dataset by its class name.

    Args:
        dataset_name (str): Name of the dataset class (e.g., "GPQA", "MedXpertQA_Skeletal").
        size (int): Dataset size to pass into the dataset constructor.

    Returns:
        List[str]: A list of problem IDs.
    """
    dataset_registry = load_dataset_classes()

    all_problems = []

    # TODO: Add problem-synthetic problem mappings.
    for d in dataset_registry.values():
        try:
            all_problems.extend(d(size=100000).problems)
        except Exception as e:
            print("Error loading problems from dataset:", d.__name__, e)
    # Synthetic mappings can be constructed by callers if needed and passed
    # through ExperimentSettings.extra; we do not mutate any global state
    # here in the refactored package.

    if dataset_name not in dataset_registry:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. "
                         f"Available datasets: {list(dataset_registry.keys())}")

    dataset_class = dataset_registry[dataset_name]
    dataset = dataset_class(size=size)

    return [problem.id for problem in dataset.problems]


if __name__ == "__main__":
    datasets = load_dataset_classes()
    print("Discovered dataset classes:")
    for name, cls in datasets.items():
        print(f"- {name}: {cls}")

    print(load_problem_ids("MedXpertQA_Skeletal"))