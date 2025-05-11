import argparse
import os

import json
from typing import List, Dict

import tqdm
import torch
import numpy as np

from mini_coil.data_pipeline.stopwords import english_stopwords
from mini_coil.data_pipeline.vocab_resolver import VocabResolver
from mini_coil.model.encoder import Encoder
from mini_coil.model.word_encoder import WordEncoder


def load_vocab(vocab_path) -> Dict[str, List[str]]:
    with open(vocab_path) as f:
        vocab = json.load(f)
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str)
    parser.add_argument("--vocab-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=4)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab_path)
    filtered_vocab = []

    for word in vocab.keys():
        if word in english_stopwords:
            continue
        model_path = os.path.join(args.models_dir, f"model-{word}.npy.npz")
        if os.path.exists(model_path):
            filtered_vocab.append(word)


    weights = [
        [],
        [],
        [],
    ]
    
    vocab_resolver = VocabResolver()

    for word in tqdm.tqdm(filtered_vocab):
        model_path = os.path.join(args.models_dir, f"model-{word}.npy.npz")

        with np.load(model_path) as data:
            weights[0].append(data['encoder_weights_0'])
            weights[1].append(data['encoder_weights_1'])
            weights[2].append(data['encoder_weights_2'])

        vocab_resolver.add_word(word)

    vocab_size = vocab_resolver.vocab_size()

    # Prepend zero weights, as first word is vocab starts from 1
    weights[0].insert(0, np.zeros(weights[0][0].shape))
    weights[1].insert(0, np.zeros(weights[1][0].shape))
    weights[2].insert(0, np.zeros(weights[2][0].shape))


    stacked_weights = [
        np.stack(weights[0], axis=0),
        np.stack(weights[1], axis=0),
        np.stack(weights[2], axis=0),
    ]

    print("stacked_weights", stacked_weights[0].shape)
    print("stacked_weights", stacked_weights[1].shape)
    print("stacked_weights", stacked_weights[2].shape)


    print("vocab_size", vocab_size)

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Save numpy file as well
    np.savez(
        args.output_path,
        encoder_weights_0=stacked_weights[0],
        encoder_weights_1=stacked_weights[1],
        encoder_weights_2=stacked_weights[2],
    )

    vocab_resolver.save_json_vocab(args.output_path + ".vocab")


if __name__ == '__main__':
    main()
