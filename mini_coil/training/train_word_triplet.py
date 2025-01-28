import argparse
import os
from typing import Tuple

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger

from mini_coil.model.word_encoder import WordEncoder
from mini_coil.training.triplet_dataloader import TripletDataloader
from mini_coil.training.triplet_word_module import TripletWordModule


def get_encoder(input_dim, output_dim, dropout: float = 0.05):
    return WordEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout
    )


def split_train_val(
        embeddings: np.ndarray,
        similarity_matrix: np.ndarray,
        line_numbers: np.ndarray,
        batch_size: int = 32,
        val_size=0.2,
) -> Tuple[TripletDataloader, TripletDataloader]:
    from_train = 0
    total_size = min(similarity_matrix.shape[0], len(line_numbers))
    from_val = int(total_size * (1 - val_size))
    to_train = from_val
    to_val = total_size

    train_dataloader = TripletDataloader(
        embeddings=embeddings,
        line_numbers=line_numbers,
        similarity_matrix=similarity_matrix,
        range_from=from_train,
        range_to=to_train,
        batch_size=batch_size,
        epoch_size=64_000,
        min_margin=0.1
    )

    val_dataloader = TripletDataloader(
        embeddings=embeddings,
        line_numbers=line_numbers,
        similarity_matrix=similarity_matrix,
        range_from=from_val,
        range_to=to_val,
        batch_size=batch_size,
        epoch_size=6400,
        min_margin=0.1
    )

    return train_dataloader, val_dataloader



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=str)
    parser.add_argument("--distance-matrix-path", type=str)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--line-numbers-path", type=str)

    args = parser.parse_args()

    embedding = np.load(args.embedding_path)

    line_numbers_path = args.line_numbers_path
    if not os.path.exists(line_numbers_path):
        raise FileNotFoundError(f"Expected line numbers file: {line_numbers_path}")
    line_numbers = np.load(line_numbers_path)

    distance_matrix = np.load(args.distance_matrix_path)

    train_loader, valid_loader = split_train_val(
        embeddings=embedding,
        similarity_matrix=distance_matrix,
        line_numbers=line_numbers,
        val_size=args.val_size,
        batch_size=args.batch_size
    )

    input_dim = embedding.shape[1]
    output_dim = args.output_dim

    encoder_load = get_encoder(input_dim, output_dim, dropout=args.dropout)

    encoder_prepared = encoder_load

    accelerator = 'cpu'
    torch.set_num_threads(1)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        enable_checkpointing=False,
        logger=CSVLogger(args.log_dir),
        # logger=TensorBoardLogger(args.log_dir),
        enable_progress_bar=True,
        accelerator=accelerator,
    )

    # with launch_ipdb_on_exception():
    trainer.fit(
        model=TripletWordModule(
            encoder_prepared,
            lr=args.lr,
            factor=args.factor,
            patience=args.patience),
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    output_dir = os.path.dirname(args.output_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.save(encoder_prepared.state_dict(), args.output_path)

    # Try to read the saved model
    encoder_load = get_encoder(input_dim, output_dim)
    encoder_load.load_state_dict(torch.load(args.output_path, weights_only=True))


if __name__ == "__main__":
    main()
