import json
import os
import argparse
from typing import List

from qdrant_client import QdrantClient, models

from mini_coil.settings import DATA_DIR

DEFAULT_SAMPLE_SIZE = 1000

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:80")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

def query_sentences(
        collection_name: str,
        words: List[str],
        sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> List[str]:

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=1000,
    )

    response = qdrant.query_points(
        collection_name=collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        query_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="sentence",
                    match=models.MatchText(text=word)
                ) for word in words
            ]
        ),
        limit=sample_size,
        with_payload=True,
        with_vectors=False,
    )

    return [point.payload["sentence"] for point in response.points]


def main():
    default_vocab_path = os.path.join(DATA_DIR, "30k-vocab-filtered.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--word", type=str)
    parser.add_argument("--collection-name", type=str, default="coil-validation")
    parser.add_argument("--output-sentences", type=str)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--vocab-path", type=str, default=default_vocab_path)

    args = parser.parse_args()

    vocab = json.load(open(args.vocab_path))

    if args.word not in vocab:
        print(f"WARNING: word {args.word} not found in vocab, using as is")
        forms = [args.word]
    else:
        forms = vocab[args.word]

    sentences = query_sentences(
        collection_name=args.collection_name,
        words=forms,
        sample_size=args.sample_size,
    )

    with open(args.output_sentences, "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

if __name__ == "__main__":
    main()
