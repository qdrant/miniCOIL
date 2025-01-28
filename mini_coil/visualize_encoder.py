import argparse
import os

import numpy as np
import torch

from mini_coil.data_pipeline.encode_and_filter import encode_and_filter
from mini_coil.training.train_word import get_encoder


def plot_embeddings(
        embeddings,
        special_point_x: float,
        special_point_y: float,
        save_path: str
):
    """
    Plot scatter plot and also one additional red dot at special point
    """
    import matplotlib.pyplot as plt

    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=1)
    plt.scatter(special_point_x, special_point_y, color='red')
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", type=str)
    parser.add_argument("--embedding-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--word", type=str)
    args = parser.parse_args()

    model_name = "jinaai/jina-embeddings-v2-small-en-tokens"
    word = args.word

    sentences = [
        "The bat flew out of the cave at dusk, its wings silhouetted against the twilight sky.",
        "A small bat darted through the trees, barely visible in the moonlight",
        "Bat dung has been mined as guano from caves and used as fertiliser.",
        "pokemon bat is a flying type pokemon.",
        "She swung the baseball bat with precision, hitting the ball right out of the park.",
        "He gripped the cricket bat tightly, ready to face the next ball.",
        "She didn’t even bat an eyelash when he told her the shocking news.",
        "She didn't bat an eyelash when he announced the surprise, keeping her composure.",
        "in the batman comics, the bat was a superhero.",
    ]

    embeddings = np.array(list(encode_and_filter(
        model_name=model_name,
        word=word,
        docs=sentences
    )))

    print(embeddings.shape)

    encoder = get_encoder(512, 4)

    encoder.load_state_dict(torch.load(args.encoder_path, weights_only=True))

    encoder.eval()

    with torch.no_grad():
        encoded = encoder(torch.from_numpy(embeddings).float())
        encoded = encoded.cpu().numpy()

    for enc, sentence in zip(encoded.tolist(), sentences):
        print(enc, sentence)

    embeddings = np.load(args.embedding_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    embeddings_x = embeddings[:, 0]
    embeddings_y = embeddings[:, 1]

    # Normalize the embeddings to the length of 1

    length = np.sqrt(embeddings_x ** 2 + embeddings_y ** 2)

    # embeddings_x /= length
    # embeddings_y /= length

    for i in range(encoded.shape[0]):
        plot_embeddings(
            embeddings=np.column_stack([embeddings_x, embeddings_y]),
            special_point_x=encoded[i, 0],
            special_point_y=encoded[i, 1],
            save_path=os.path.join(args.output_dir, f"encoded_{i}.png")
        )


if __name__ == "__main__":
    main()
