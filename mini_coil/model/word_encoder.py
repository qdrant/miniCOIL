"""
Encoder model for a single word.

This is an intermediate form of the model, that is designed to be trained independently for each word.
The model is a simple linear encoder, that compresses the input token embeddings into a smaller representation.

Additionally, it includes layers that simulate quantization into int8.

After training, all word encoders are combined into a single model defined in `encoder.py`.
"""

import math

import torch

from torch import nn
from torch.nn import init


class WordEncoder(nn.Module):
    """
        WordEncoder(512, 4)

        Will look like this:


                                      Linear transformation
         ┌─────────────────────┐      ┌─────────┐    ┌─────────┐
         │ Token Embedding(512)├─────►│512->4   ├───►│Tanh     ├──► 4d representation
         └─────────────────────┘      └─────────┘    └─────────┘


         Final liner transformation is accompanied by a non-linear activation function: Sigmoid.

         Tanh is used to ensure that the output is in the range [-1, 1].
         It would be easier to visually interpret the output of the model, assuming that each dimension
         would need to encode a type of semantic cluster.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            device=None,
            dtype=None,
            dropout: float = 0.05
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype

        self.dropout = nn.Dropout(dropout)

        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()

        self.encoder_weights = nn.Parameter(torch.zeros((input_dim, output_dim), **factory_kwargs))

        self.activation = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.encoder_weights, a=math.sqrt(5))

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            word_embeddings: (batch_size, input_dim) - input token embeddings

        Returns:
            (batch_size, output_dim) - compressed representation of the input
        """
        # word_embeddings = self.quant(word_embeddings)
        word_embeddings = self.dropout(word_embeddings)
        compressed = self.activation(word_embeddings @ self.encoder_weights)
        return compressed
        # return self.dequant(compressed)


