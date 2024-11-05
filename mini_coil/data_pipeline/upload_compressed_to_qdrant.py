import argparse
import os
from typing import Iterable
import json

from qdrant_client import QdrantClient, models
import numpy as np
import hashlib
import tqdm

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


def load_sentences(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-path", type=str, default=None)
    parser.add_argument("--compressed-path", type=str)
    parser.add_argument("--collection-name", type=str, default="minicoil")
    parser.add_argument("--recreate-collection", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    vectors = np.load(args.compressed_path)

    dim = vectors.shape[1]

    collection_name = args.collection_name

    collection_exists = client.collection_exists(collection_name)

    if collection_exists and args.recreate_collection:
        client.delete_collection(collection_name)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )

    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=load_sentences(args.sentences_path),
    )


if __name__ == "__main__":
    main()
