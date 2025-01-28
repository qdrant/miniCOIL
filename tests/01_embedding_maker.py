import argparse

import numpy as np
import tqdm
from fastembed.text import TextEmbedding
from sentence_transformers import SentenceTransformer

from mini_coil.model.mini_coil_inference import MiniCOIL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file containing sentences")
    parser.add_argument("--output-mixedbread", type=str, required=False, help="Path to the output file for mixedbread")
    parser.add_argument("--output-jina", type=str, required=False, help="Path to the output file for jina2")
    parser.add_argument("--output-minicoil", type=str, required=False, help="Path to the output file for minicoil")
    parser.add_argument("--output-random", type=str, required=False, help="Path to the output file for random")

    parser.add_argument("--vocab-path", type=str, required=True, help="Path to the vocabulary file (minicoil)")
    parser.add_argument("--word-encoder-path", type=str, required=True, help="Path to the word encoder file (minicoil)")
    parser.add_argument("--use-cuda", action="store_true", default=False, help="Use CUDA for jina2base")
    parser.add_argument("--minicoil-test-word", type=str, required=True, help="Word to test for minicoil")

    parser.add_argument("--dim", type=int, default=4, help="Output dimension for minicoil")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        lines = [x.strip() for x in f if len(x.strip()) > 0]

    skip_mixed_bread = args.output_mixedbread is None
    skip_jina_small = args.output_jina is None
    skip_minicoil = args.output_minicoil is None
    skip_random = args.output_random is None

    if not skip_mixed_bread:
        model_mixed = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True)
        if args.use_cuda:
            model_mixed = model_mixed.to("cuda")
        emb_mixed = model_mixed.encode(lines, batch_size=32, show_progress_bar=True)
        np.save(args.output_mixedbread, emb_mixed)
        del model_mixed

    if not skip_jina_small:
        model_jina = TextEmbedding("jinaai/jina-embeddings-v2-small-en", cuda=args.use_cuda)
        emb_jina = np.stack(list(model_jina.embed(tqdm.tqdm(lines), batch_size=8)))
        np.save(args.output_jina, emb_jina)
        del model_jina

    if not skip_random:
        emb_random = np.random.randn(len(lines), 768)
        np.save(args.output_random, emb_random)

    if not skip_minicoil:
        model_minicoil = MiniCOIL(
            vocab_path=args.vocab_path,
            word_encoder_path=args.word_encoder_path,
            sentence_encoder_model="jinaai/jina-embeddings-v2-small-en-tokens",
        )
        emb_mc_list = []
        for line in lines:
            row = model_minicoil.encode([line])[0]
            v = []
            if args.minicoil_test_word not in row:
                zeros = np.zeros((model_minicoil.output_dim,))
                emb_mc_list.append(zeros)
            else:
                emb_mc_list.append(row[args.minicoil_test_word]["embedding"])
        emb_mc = np.stack(emb_mc_list)
        np.save(args.output_minicoil, emb_mc)


if __name__ == "__main__":
    main()
