import argparse
import os
import numpy as np

from tensorflow import keras
from keras import ops
import tensorflow as tf
from mini_coil.explore.custom_parametric_umap import ParametricUMAP


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def load_embeddings(embedding_path: str) -> np.ndarray:
    return np.load(embedding_path)


def make_one_layer_model(input_dim, output_dim):

    # Total number of parameters: 
    # 
    # input_dim * 8 + 8 * 16 + 16 * output_dim
    # 
    # Example: input_dim = 512, output_dim = 4
    # 512 * 8 + 8 * 16 + 16 * 4 = 4096 + 128 + 64 = 4288 Floats
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(8, activation='relu', use_bias=False),
        keras.layers.Dense(16, activation='relu', use_bias=False),
        keras.layers.Dense(output_dim, activation=None, use_bias=False)
    ])



def train_parametric_umap(embeddings: np.ndarray, output_dim: int, output_path: str, output_embeddings_path: str):

    input_dim = embeddings.shape[1]
    output_dim_umap = output_dim - 1

    # 2. Custom 1-layer model
    parametric_umap = ParametricUMAP(
        n_components=output_dim_umap,
        encoder=make_one_layer_model(input_dim, output_dim_umap),
        decoder=None
    )

    parametric_umap.fit(embeddings)

    train_embeddings = parametric_umap.transform(embeddings)

    # Make sure directory exists
    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_embeddings_path, train_embeddings)

    np.savez(output_path,
        encoder_weights_0=parametric_umap.encoder.weights[0].value.numpy(),
        encoder_weights_1=parametric_umap.encoder.weights[1].value.numpy(),
        encoder_weights_2=parametric_umap.encoder.weights[2].value.numpy(),
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=str)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument('--output-model-path', type=str)
    parser.add_argument('--output-embeddings-path', type=str)

    args = parser.parse_args()

    embeddings = load_embeddings(args.embedding_path)
    train_parametric_umap(embeddings, args.output_dim, args.output_model_path, args.output_embeddings_path)


if __name__ == "__main__":
    main()
