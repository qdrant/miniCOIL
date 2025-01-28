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
        model_path = os.path.join(args.models_dir, f"model-{word}.ptch")
        if os.path.exists(model_path):
            filtered_vocab.append(word)

    params = [torch.zeros(args.input_dim, args.output_dim)]  # Extra zero tensor, as first word is vocab starts from 1

    vocab_resolver = VocabResolver()

    for word in tqdm.tqdm(filtered_vocab):
        model_path = os.path.join(args.models_dir, f"model-{word}.ptch")
        encoder = WordEncoder(args.input_dim, args.output_dim)
        encoder.load_state_dict(torch.load(model_path, weights_only=True))

        encode_param = encoder.encoder_weights.data
        params.append(encode_param)

        vocab_resolver.add_word(word)

    vocab_size = vocab_resolver.vocab_size()

    combined_params = torch.stack(params, dim=0)

    print("combined_params", combined_params.shape)
    print("vocab_size", vocab_size)

    combined_params_numpy = combined_params.numpy()

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Save numpy file as well
    np.save(args.output_path + ".npy", combined_params_numpy)

    encoder = Encoder(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        vocab_size=vocab_size,
    )

    encoder.encoder_weights.data = combined_params

    torch.save(encoder.state_dict(), args.output_path)

    vocab_resolver.save_json_vocab(args.output_path + ".vocab")


if __name__ == '__main__':
    main()
