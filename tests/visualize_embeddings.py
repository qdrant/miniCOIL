import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    """
    Take numpy file with produced embeddings and visualize first 2 dimensions.

    Make big plot to see the distribution of the embeddings.

    In addition to scatterplot, print histogram of circular distribution of the embeddings around 0,0.
    Take the angle of the embedding and plot histogram of the angles.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input numpy file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output visualization")
    args = parser.parse_args()

    emb = np.load(args.input)

    plt.figure(figsize=(10, 10))
    plt.scatter(emb[:, 0], emb[:, 1])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    plt.savefig(args.output + ".png")

    print(f"Visualization saved to {args.output}")

    embedding_angles = np.arctan2(emb[:, 1], emb[:, 0])

    plt.figure(figsize=(10, 10))
    plt.hist(embedding_angles, bins=100)

    plt.savefig(args.output + "_histogram.png")

    print(f"Histogram saved to {args.output}_histogram.png")

if __name__ == "__main__":
    main()

