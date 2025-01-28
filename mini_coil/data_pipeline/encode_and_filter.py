import argparse
import json
import os
from typing import List, Iterable, Optional, Dict

import numpy as np
import tqdm
from fastembed.late_interaction.token_embeddings import TokenEmbeddingsModel
from npy_append_array import NpyAppendArray

from mini_coil.data_pipeline.vocab_resolver import VocabResolver, VocabTokenizerTokenizer


def load_model(model_name):
    return TokenEmbeddingsModel(model_name=model_name, threads=1)


def read_sentences(file_path: str, limit_length: int = 4096) -> Iterable[Dict[str, str]]:
    with open(file_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            yield {
                "sentence": doc["sentence"][:limit_length],
                "line_number": doc["line_number"],
            }


def encode_and_filter(model_name: Optional[str], word: str, docs: List[dict]) -> Iterable[np.ndarray]:
    model = load_model(model_name)
    vocab_resolver = VocabResolver(tokenizer=VocabTokenizerTokenizer(model.tokenizer))
    vocab_resolver.add_word(word)

    sentences = [doc["sentence"] for doc in docs]

    for embedding, sentence in zip(model.embed(sentences, batch_size=2), sentences):
        token_ids = np.array(model.tokenize([sentence])[0].ids)
        word_mask, counts, oov, _forms = vocab_resolver.resolve_tokens(token_ids)
        word_mask = word_mask.astype(bool)
        total_tokens = np.sum(word_mask)
        if total_tokens == 0:
            yield np.zeros(embedding.shape[1])
            continue

        word_embeddings = embedding[word_mask]
        avg_embedding = np.mean(word_embeddings, axis=0)
        yield avg_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--output-line-numbers-file", type=str)
    parser.add_argument("--word", type=str)
    args = parser.parse_args()

    model_name = "jinaai/jina-embeddings-v2-small-en-tokens"

    input_file = args.sentences_file
    docs = list(read_sentences(input_file, limit_length=1024))

    embeddings = encode_and_filter(
        model_name=model_name,
        word=args.word,
        docs=docs
    )

    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    text_np_emb_file = NpyAppendArray(output_file, delete_if_exists=True)
    line_numbers_file = args.output_line_numbers_file
    line_numbers = []

    for doc, emb in tqdm.tqdm(zip(docs, embeddings), total=len(docs)):
        emb_conv = emb.reshape(1, -1)
        text_np_emb_file.append(emb_conv)
        line_numbers.append(int(doc["line_number"]))

    text_np_emb_file.close()
    np.save(line_numbers_file, np.array(line_numbers))

    text_np_emb_file = np.load(output_file, mmap_mode='r')
    print(f"text_np_emb_file {output_file} shape:", text_np_emb_file.shape)
    print(f"line_numbers {line_numbers_file} shape:", np.load(line_numbers_file).shape)


if __name__ == "__main__":
    main()
