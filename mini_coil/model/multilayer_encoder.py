"""
Pure numpy implementation of encoder model for a single word.

This model is not trainable, and should only be used for inference.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple


class MultiLayerEncoder:
    """
    Pure numpy implementation of 

    ```python
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(4, activation='relu', use_bias=False),
        layers.Dense(16, activation='relu', use_bias=False),
        layers.Dense(output_dim, activation=None, use_bias=False)
    ```

    This model is not trainable, and should only be used for inference.
    """

    @classmethod
    def relu(cls, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.maximum(0, x)
    
    @classmethod
    def convert_to_sphere(cls, x: NDArray[np.float32], radius: float = 25.0) -> NDArray[np.float32]:
        """
        Put N-dimensional embeddings into a sphere of radius r.
        Effectively, this generates a N+1-dimensional embedding where the last dimension is the radius.

        WARN: Radius should be greater than the maximum norm of the input embeddings.
        
        Args:
            x: (batch_size, input_dim) float array

        Returns:
            (batch_size, input_dim + 1) float array
        """

        norms = np.linalg.norm(x, axis=1)

        max_norm = np.max(norms)

        if max_norm > radius:
            raise ValueError(f"Maximum norm of input embeddings is greater than the radius. {max_norm} > {radius}")

        t = np.sqrt(radius**2 - norms**2).reshape(-1, 1)

        sphere_coords = np.hstack([x, t])

        return sphere_coords

    def __init__(
        self,
        weights: List[NDArray[np.float32]],
        make_sphere: bool = False,
    ):
        
        vocab_sizes = []
        for weight in weights:
            vocab_sizes.append(weight.shape[0])

        vocab_sizes = list(set(vocab_sizes))

        assert len(vocab_sizes) == 1, "All weights must have the same vocab size"

        self.vocab_size = vocab_sizes[0]
        self.input_dim = weights[0].shape[1]

        if make_sphere:
            self.output_dim = weights[-1].shape[2] + 1
        else:
            self.output_dim = weights[-1].shape[2]

        self.encoder_weights: List[NDArray[np.float32]] = weights

        # Activation function
        self.activation = MultiLayerEncoder.relu

        self.make_sphere = make_sphere

    @staticmethod
    def convert_vocab_ids(vocab_ids: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Convert vocab_ids of shape (batch_size, seq_len) into (batch_size, seq_len, 2)
        by appending batch_id alongside each vocab_id.
        """
        batch_size, seq_len = vocab_ids.shape
        batch_ids = np.arange(batch_size, dtype=vocab_ids.dtype).reshape(batch_size, 1)
        batch_ids = np.repeat(batch_ids, seq_len, axis=1)
        # Stack vocab_ids and batch_ids along the last dimension
        combined: NDArray[np.float32] = np.stack((vocab_ids, batch_ids), axis=2).astype(np.int32)
        return combined

    @classmethod
    def avg_by_vocab_ids(
        cls, vocab_ids: NDArray[np.int32], embeddings: NDArray[np.float32]
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Takes:
            vocab_ids: (batch_size, seq_len) int array
            embeddings: (batch_size, seq_len, input_dim) float array

        Returns:
            unique_flattened_vocab_ids: (total_unique, 2) array of [vocab_id, batch_id]
            unique_flattened_embeddings: (total_unique, input_dim) averaged embeddings
        """
        input_dim = embeddings.shape[2]

        # Flatten vocab_ids and embeddings
        # flattened_vocab_ids: (batch_size*seq_len, 2)
        flattened_vocab_ids = cls.convert_vocab_ids(vocab_ids).reshape(-1, 2)

        # flattened_embeddings: (batch_size*seq_len, input_dim)
        flattened_embeddings = embeddings.reshape(-1, input_dim)

        # Find unique (vocab_id, batch_id) pairs
        unique_flattened_vocab_ids, inverse_indices = np.unique(
            flattened_vocab_ids, axis=0, return_inverse=True
        )

        # Prepare arrays to accumulate sums
        unique_count = unique_flattened_vocab_ids.shape[0]
        unique_flattened_embeddings = np.zeros((unique_count, input_dim), dtype=np.float32)
        unique_flattened_count = np.zeros(unique_count, dtype=np.int32)

        # Use np.add.at to accumulate sums based on inverse indices
        np.add.at(unique_flattened_embeddings, inverse_indices, flattened_embeddings)
        np.add.at(unique_flattened_count, inverse_indices, 1)

        # Compute averages
        unique_flattened_embeddings /= unique_flattened_count[:, None]

        return unique_flattened_vocab_ids.astype(np.int32), unique_flattened_embeddings.astype(
            np.float32
        )

    def forward(
        self, vocab_ids: NDArray[np.int32], embeddings: NDArray[np.float32]
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Args:
            vocab_ids: (batch_size, seq_len) int array
            embeddings: (batch_size, seq_len, input_dim) float array

        Returns:
            unique_flattened_vocab_ids_and_batch_ids: (total_unique, 2)
            unique_flattened_encoded: (total_unique, output_dim)
        """
        # Average embeddings for duplicate vocab_ids
        unique_flattened_vocab_ids_and_batch_ids, unique_flattened_embeddings = (
            self.avg_by_vocab_ids(vocab_ids, embeddings)
        )

        # Select the encoder weights for each unique vocab_id
        unique_flattened_vocab_ids = unique_flattened_vocab_ids_and_batch_ids[:, 0].astype(
            np.int32
        )
        
        transformed_embeddings = unique_flattened_embeddings

        for i, weight in enumerate(self.encoder_weights):

            # unique_encoder_weights: (total_unique, input_dim, output_dim)
            unique_encoder_weights = weight[unique_flattened_vocab_ids]

            # Compute linear transform: (total_unique, output_dim)
            # Using Einstein summation for matrix multiplication:
            # 'bi,bio->bo' means: for each "b" (batch element), multiply embeddings (b,i) by weights (b,i,o) -> (b,o)
            transformed_embeddings = np.einsum(
                "bi,bio->bo", transformed_embeddings, unique_encoder_weights
            )
            
            if i != len(self.encoder_weights) - 1:
                # Apply activation on all but the last layer
                transformed_embeddings = self.activation(transformed_embeddings).astype(np.float32)

        if self.make_sphere:
            transformed_embeddings = self.convert_to_sphere(transformed_embeddings)

        return unique_flattened_vocab_ids_and_batch_ids.astype(np.int32), transformed_embeddings
