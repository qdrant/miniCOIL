import torch
from torch import nn


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor, eps=1e-8):
    """
    Calculate pairwise cosine distance between two tensors.

    Args:
        x1: (batch_size, embedding_size) - embeddings of the first elements
        x2: (batch_size, embedding_size) - embeddings of the second elements
        eps: float - small value to avoid division by zero

    Returns:
        distance: (batch_size) - cosine distance between the elements
    """

    dot_product = torch.sum(x1 * x2, dim=1)
    x1_norm = torch.norm(x1, p=2, dim=1)
    x2_norm = torch.norm(x2, p=2, dim=1)

    return 1.0 - dot_product / (x1_norm * x2_norm + eps)


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            embeddings: torch.Tensor,
            triplet_indices: torch.Tensor,
            margins: torch.Tensor
    ):
        """
        Calculate the triplet loss with custom margin for each triplet.

        Args:
            embeddings: (batch_size, embedding_size) - embeddings of the elements
            triplet_indices: (num_triplets, 3) - indices of the triplets. Each row consists of: anchor, positive, negative
            margins: (num_triplets) - margins for each triplet

        Returns:
            loss: triplet loss for the batch
        """

        # (num_triplets, embedding_size)
        anchor = embeddings[triplet_indices[:, 0]]
        # (num_triplets, embedding_size)
        positive = embeddings[triplet_indices[:, 1]]
        # (num_triplets, embedding_size)
        negative = embeddings[triplet_indices[:, 2]]

        # (num_triplets)
        positive_distance = cosine_distance(anchor, positive)
        # (num_triplets)
        negative_distance = cosine_distance(anchor, negative)

        # (num_triplets)
        # Example:
        # positive_distance = [0.1, 0.2, 0.5]
        # negative_distance = [0.3, 0.3, 0.2]
        # margins           = [0.2, 0.2, 0.3]
        # loss              = [
        #                       0.1 - 0.3 + 0.2 = 0.0,
        #                       0.2 - 0.3 + 0.2 = 0.1,
        #                       0.5 - 0.2 + 0.3 = 0.6
        #                     ]
        loss = torch.relu(positive_distance - negative_distance + margins)
        number_failed_triplets = torch.sum(loss > 0).item()
        return loss.mean(), number_failed_triplets

def test_cosine_distance():
    x1 = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ])
    x2 = torch.tensor([
        [0.3, 0.2, 0.1],
        [3.0, 2.0, 1.0],
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.0],
    ])

    distance = cosine_distance(x1, x2)

    print(distance)

    assert distance[0] > distance[2]
    assert distance[0] - distance[1] < 1e-6
    assert distance[1] > distance[2]
    assert distance[3] > distance[2]


if __name__ == "__main__":
    test_cosine_distance()
