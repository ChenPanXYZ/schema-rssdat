import pickle
import numpy as np
from src.entity.problems.Problem import Problem


def LoadEmbeddingUseCase(knowledge_base, embedding_path) -> [Problem]:
    """
    Load item embeddings from the embeddings directory.
    """
    # TODO: Very important! Need to make sure the order matches the order of the problems in the knowledge base!
    # embedding_path_list = [os.path.join(embedding_path, file_name) for file_name in os.listdir(
    #     embedding_path) if file_name.endswith(".pkl")]
    #
    # embedding_path_list = sorted(embedding_path_list, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    # embedding_list = []
    embedding_list = []
    for problem in knowledge_base.knowledges:
        with open(f'{embedding_path}/{problem.id}.pkl', "rb") as file:
            embeddings = pickle.load(file)
            embedding_list.append(embeddings)

    return np.array(embedding_list)

