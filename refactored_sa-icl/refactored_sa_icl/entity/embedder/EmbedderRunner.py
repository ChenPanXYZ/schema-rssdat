from src.entity.embedder.STEmbedder import STEmbedder
from src.entity.embedder.GPTEmbedder import GPTEmbedder

from src.entity.datasets.Dataset import Dataset
from src.entity.datasets.GPQA import GPQA


import argparse

API_KEY=""

EMBEDDER_TYPES = {"gpt", "st"}
MODES = {"question", "question", "question + answer"}

def start_embedding(dataset:Dataset, model_name, emb_type, mode, output_dir):
    
    if emb_type == "gpt":
        embedder = GPTEmbedder(api_key=API_KEY)
    problems = dataset.problems
    if emb_type == "st":
        embedder = STEmbedder()
    else:
        raise ValueError("Invalid embedder type. Available types are: {}".format(", ".join(sorted(EMBEDDER_TYPES))))
    embedder.create_embeddings(problems=problems, mode=mode, output_dir=output_dir)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for text files using either a GPT model or a sentence transformer.")
    parser.add_argument("-m", "--mode", type=str, choices=MODES, required=True, help="Mode of the dataset to generate embeddings for.")
    parser.add_argument("-s", "--size", type=str, required=True, help="Size of the dataset to generate embeddings for.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path where the generated embeddings should be saved.")
    parser.add_argument("--emb_type", type=str, choices=EMBEDDER_TYPES, default="gpt",
                        help="Type of embedder to use. Available options: {}".format(", ".join(sorted(EMBEDDER_TYPES))))
    args = parser.parse_args()
    
    dataset = GPQA(size=args.size)
    start_embedding(dataset=dataset, emb_type=args.emb_type, mode=args.mode, output_dir=args.output_dir)