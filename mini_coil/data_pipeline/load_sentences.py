import argparse
import json
import os
from os import getenv
from typing import List

from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", getenv("QDRANT_URL", "http://localhost:80"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", getenv("QDRANT_API_KEY", ""))

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    port=80,
    timeout=999,
    https=False,
)


# Iterate over vocabulary and distance matrixes to select which abstracts and sentences should we used for inference
# Additionally save the words accosiated with the selected abstracts and sentences

def load_vocabulary(path: str) -> List[str]:
    vocabulary = []
    with open(path, 'r') as f:
        for line in f:
            vocabulary.append(line.strip())

    return vocabulary


def load_matrix_ids(directory: str, word: str):
    path = os.path.join(directory, f"sparse_matrix_{word}.json")
    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        matrix = json.load(f)
        ids = matrix['ids']
        return ids


def load_sentences(collection_name: str, ids: List[str]) -> List[dict]:
    points_to_abstracts = {}
    batch_size = 10000

    for i in range(0, len(ids), batch_size):
        print("Retrieving batch", i + 1, "of", len(ids) // batch_size)
        batch_ids = ids[i:i + batch_size]
        points = client.retrieve(
            collection_name,
            batch_ids,
            with_payload=True,
            with_vectors=False,
        )
        batch_points = dict((point.id, point.payload) for point in points)
        points_to_abstracts.update(batch_points)

    result = []
    for point_id in ids:
        if point_id not in points_to_abstracts:
            print(f"Point {point_id} not found in collection {collection_name}")
            exit(1)

        payload = points_to_abstracts[point_id]
        ln = payload["line_number"]

        result.append({
            "id": point_id,
            "line_number": ln,
            "sentence": payload["sentence"],
            "abs_hash": payload["abs_hash"]
        })

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstracts-collection-name", type=str, default="coil-abstracts")
    parser.add_argument("--sentences-collection-name", type=str, default="coil")
    parser.add_argument("--word", type=str)
    parser.add_argument("--matrix-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    sentences_collection_name = args.sentences_collection_name
    abstracts_collection_name = args.abstracts_collection_name

    word = args.word

    ids = load_matrix_ids(args.matrix_dir, word)
    if ids is None:
        print(f"Matrix for word {word} does not exist")
        return

    sentences_data = load_sentences(sentences_collection_name, ids)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"sentences-{word}.jsonl")

    with open(output_path, 'w') as f:
        for sentence in sentences_data:
            f.write(json.dumps(sentence) + '\n')


if __name__ == "__main__":
    main()
