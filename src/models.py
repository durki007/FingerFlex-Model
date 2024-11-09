import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .raw_models import AutoEncoder1D

from .tools import correlation_metric


class BaseEcogFingerflexModel(pl.LightningModule):
    """
        The class which encapsulates the model, its optimizer and the training process at different stages, including logging
    """

    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.model = model  # Pytorch model
        self.lr = 8.42e-5

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = F.mse_loss(y_hat, y)
        corr = correlation_metric(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"cosine_dst_train", corr, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return 0.5 * loss + 0.5 * (1. - corr)  # возврат значения функции потерь

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)

        corr = correlation_metric(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("cosine_dst_val", corr, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return y_hat  # Return the result for the validation callback

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     weight_decay=1e-6)  # set optimizer, lr and L2 regularization coeff
        return optimizer

