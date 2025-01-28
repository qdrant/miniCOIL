import lightning as L
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mini_coil.model.triplet_loss import TripletLoss
from mini_coil.model.word_encoder import WordEncoder


class TripletWordModule(L.LightningModule):
    def __init__(self, encoder: WordEncoder, lr: float = 2e-3, factor: float = 0.5, patience: int = 5):
        super().__init__()
        self.encoder = encoder
        self.loss = TripletLoss()
        self.lr = lr
        self.factor = factor
        self.patience = patience

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.factor,
                    patience=self.patience,
                    verbose=True,
                    threshold=1e-4
                ),
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        embeddings = torch.from_numpy(batch["embeddings"]).float().to(self.device)
        triplets = torch.from_numpy(batch["triplets"]).long().to(self.device)
        margins = torch.from_numpy(batch["margins"]).float().to(self.device)
        encoded = self.encoder(embeddings)
        loss_val, _ = self.loss(
            encoded,
            triplets,
            margins,
        )
        self.log(
            "train_loss",
            loss_val,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            batch_size=embeddings.size(0)
        )
        return loss_val

    def validation_step(self, batch, batch_idx):
        embeddings = torch.from_numpy(batch["embeddings"]).float().to(self.device)
        triplets = torch.from_numpy(batch["triplets"]).long().to(self.device)
        margins = torch.from_numpy(batch["margins"]).float().to(self.device)
        encoded = self.encoder(embeddings)
        loss_val, number_failed_triplets = self.loss(
            encoded,
            triplets,
            margins,
        )
        self.log(
            "val_loss",
            loss_val,
            batch_size=embeddings.size(0)
        )
        self.log(
            "val_failed_triplets",
            number_failed_triplets,
            batch_size=embeddings.size(0)
        )
        return loss_val
