import argparse
import os

import tqdm
import torch
import numpy as np

from mini_coil.data_pipeline.stopwords import english_stopwords
from mini_coil.data_pipeline.vocab_resolver import VocabResolver
from mini_coil.model.encoder import Encoder
from mini_coil.model.word_encoder import WordEncoder
from py_rust_stemmers import SnowballStemmer

def load_vocab(vocab_path):
    vocab = []
    with open(vocab_path, 'r') as f:
        for line in f:
            word = line.strip()
            vocab.append(word)
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str)
    parser.add_argument("--vocab-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=4)

    stemmer = SnowballStemmer("english")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab_path)
    filtered_vocab = []
    stemmed_words_vocab = set()

    for word in vocab:
        if word in english_stopwords:
            continue
        stemmed_word = stemmer.stem_word(word)
        if stemmed_word in stemmed_words_vocab:
            continue
        else:
            stemmed_words_vocab.add(stemmed_word)
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
