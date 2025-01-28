import argparse

import os
import numpy as np
import tqdm

from mini_coil.model.mini_coil_inference import MiniCOIL

ANECDOTAL_EXAMPLES = {
    "vector": [
        "vector search",
        "vector index",
        "vector space",
        "vector image",
        "vector graphics",
        "vector illustration",
    ],
    "bat": [
        "baseball bat",
        "swing a bat",
        "aluminum bat",
        "bat in the cave",
        "vampire bat",
        "bat flying",
    ],
    "calcul": [
        "I know how to use calculators",
        "open the calculator app",
        "calculating this won't be easy",
        "numerical calculations",
    ],
    "life": [
        "How much money is sufficient to live a peaceful life?",
        "How much money do we need to live life happily?",
    ]
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to the vocabulary file (minicoil)")
    parser.add_argument("--word-encoder-path", type=str, required=True, help="Path to the word encoder file (minicoil)")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file containing sentences")
    parser.add_argument("--output", type=str, required=True, help="Path to the output minicoil embeddings")
    parser.add_argument("--word", type=str, required=True, help="Word to test for minicoil")

    args = parser.parse_args()

    model_minicoil = MiniCOIL(
        vocab_path=args.vocab_path,
        word_encoder_path=args.word_encoder_path,
        sentence_encoder_model="jinaai/jina-embeddings-v2-small-en-tokens",
    )
    emb_mc_list = []

    lines = open(args.input_file).read().splitlines()

    encoded = model_minicoil.encode(tqdm.tqdm(lines), parallel=4)

    for row in encoded:
        if args.word not in row:
            zeros = np.zeros((model_minicoil.output_dim,))
            emb_mc_list.append(zeros)
        else:
            emb_mc_list.append(row[args.word]["embedding"])
    emb_mc = np.stack(emb_mc_list)

    # create output dir is not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    np.save(args.output, emb_mc)

    # If the word happens to be in the anecdotal examples, print the embeddings
    if args.word in ANECDOTAL_EXAMPLES:
        examples = ANECDOTAL_EXAMPLES[args.word]
        encoded = model_minicoil.encode(examples)
        for example, row in zip(examples, encoded):
            word_emb = row[args.word]["embedding"]
            print(word_emb, " - ", example)


if __name__ == "__main__":
    main()