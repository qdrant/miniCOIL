import argparse
import gzip
import os
from typing import Iterable
import numpy as np
from npy_append_array import NpyAppendArray
from datasets import load_dataset
from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
import torch


def load_dataset_hf(name: str, max_count=None):
    if name.lower() in ["msmarco", "msmarco_v1_1"]:
        ds = load_dataset("ms_marco", "v1.1", split="train")
        for i, ex in enumerate(ds):
            passages = ex["passages"]["passage_text"]
            for passage in passages:
                yield passage.strip().replace("\n", " ")
            if max_count and i >= max_count:
                break
    else:
        raise ValueError(f"Dataset non supporté: {name}")


def read_texts(path: str, vocab=None, max_count=None) -> Iterable[str]:
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            try:
                _id, sentence = line.strip().split("\t", 1)
            except ValueError:
                continue

            if vocab:
                tokens = set(sentence.lower().split())
                if not tokens.intersection(vocab):
                    continue

            yield sentence
            if max_count and i >= max_count:
                break


def get_model(model_name: str):
    try:
        model = TextEmbedding(model_name)
        print(f"⚡ Using FastEmbed model: {model_name}")
        return model, "fastembed"
    except Exception:
        print(f"⚡ Using SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        return model, "sentence_transformer"


def encode_texts(texts: Iterable[str], model, model_type: str, batch_size: int, use_cuda: bool):
    if model_type == "fastembed":
        for embeddings in model.embed(texts, batch_size=batch_size):
            yield embeddings
    elif model_type == "sentence_transformer":
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        print(f"✅ SentenceTransformer running on device: {device}")
        model.to(device)
        model.eval()
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        yield embeddings
    else:
        raise ValueError(f"Model type non supporté: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--wordlist", type=str)
    parser.add_argument("--model-name", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--max-count", type=int)

    args = parser.parse_args()

    # 🔹 Supprime ancien fichier pour éviter append de shapes différentes
    if os.path.exists(args.output_file):
        print(f"⚠️ Existing file found, removing: {args.output_file}")
        os.remove(args.output_file)

    vocab = None
    if args.wordlist:
        with open(args.wordlist, "r", encoding="utf-8") as f:
            vocab = set(f.read().splitlines())
        print(f"Loaded vocabulary with {len(vocab)} words.")

    if args.input_file:
        texts_generator = read_texts(args.input_file, vocab=vocab, max_count=args.max_count)
    elif args.dataset:
        texts_generator = load_dataset_hf(args.dataset, max_count=args.max_count)
    else:
        raise ValueError("Must specify either --input-file or --dataset")

    model, model_type = get_model(args.model_name)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    total_encoded = 0
    with NpyAppendArray(args.output_file) as npaa:
        texts_list = list(texts_generator)
        for embeddings_batch in encode_texts(texts_list, model, model_type, args.batch_size, args.use_cuda):
            embeddings_batch = np.array(embeddings_batch, dtype=np.float32)
            print(f"🧩 Batch shape: {embeddings_batch.shape}")
            npaa.append(embeddings_batch)
            total_encoded += embeddings_batch.shape[0]

    print(f"\n✅ Embeddings saved: {args.output_file}")
    print(f"📊 Total sentences encoded: {total_encoded}")


if __name__ == "__main__":
    main()
