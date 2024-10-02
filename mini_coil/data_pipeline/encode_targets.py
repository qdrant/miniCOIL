"""
This script takes raw training data and applies initial embeddings to it.

The output of the process, for each abstract:

* List of token ids
* List of per-token embeddings
* Aggregate embedding of the abstract

This script can potentially generate huge amounts of data, so
it will write directly to disk.

"""

import os
from typing import Iterable
import argparse

import numpy as np
import tqdm
from npy_append_array import NpyAppendArray
from sentence_transformers import SentenceTransformer

from mini_coil.data_pipeline.pre_encoder import PreEncoder
from mini_coil.settings import DATA_DIR
from mini_coil.data_pipeline.vocab_resolver import VocabResolver


def read_texts(path: str) -> Iterable[str]:
    with open(path, "r") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if len(line) > 0:
                document_hash, text = line.split("\t", 1)
                yield document_hash, text


def iter_batch(iterable, size):
    batch = []
    hashes = []
    for document_hash, item in iterable:
        batch.append(item)
        hashes.append(document_hash)
        if len(batch) >= size:
            yield hashes, batch
            batch = []
            hashes = []
    if len(batch) > 0:
        yield hashes, batch


def main():
    input_file = "bat.txt"

    default_input_data_path = os.path.join(DATA_DIR, "test", input_file)
    default_output_file = os.path.join(DATA_DIR, "test")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=default_input_data_path)
    parser.add_argument("--output-file", type=str, default=default_output_file)
    args = parser.parse_args()

    model_repository = "mixedbread-ai/mxbai-embed-large-v1"

    model = SentenceTransformer(model_repository, trust_remote_code=True)

    batch_size = 1024

    output_file = args.output_file
    output_dir = os.path.basename(output_file)

    os.makedirs(output_dir, exist_ok=True)

    npy_file = f"{output_file}.npy"
    hash_file = f"{output_file}.txt"

    text_np_emb_file = NpyAppendArray(npy_file, delete_if_exists=True)

   with open(hash_file, 'w') as hash_output:
        for hashes, batch in iter_batch(read_texts(args.input_file), batch_size):
            text_embeddings = model.encode(batch)
            text_np_emb_file.append(text_embeddings)
            
            for hash_value in hashes:
                hash_output.write(f"{hash_value}\n")

    text_np_emb_file.close()

    # Check the output file shape

    text_np_emb_file = np.load(output_file, mmap_mode='r')

    print(f"text_np_emb_file {output_file} shape:", text_np_emb_file.shape)


if __name__ == "__main__":
    main()
