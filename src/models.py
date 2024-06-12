import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools import correlation_metric


class BaseEcogFingerflexModel(pl.LightningModule):
    """
        The class which encapsulates the model, its optimizer and the training process at different stages, including logging
    """

    def __init__(self, model):
        super().__init__()
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


class ConvBlock(nn.Module):
    """
    Convolution block:
        - 1d conv
        - layer norm by embedding axis
        - activation
        - dropout
        - Max pooling
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, p_conv_drop=0.1):
        super(ConvBlock, self).__init__()

        # use it instead stride.

        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                bias=False,
                                padding='same')

        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)

        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv1d(x)

        # norm by last axis.
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)

        x = self.activation(x)

        x = self.drop(x)

        x = self.downsample(x)

        return x


class UpConvBlock(nn.Module):
    """
    Decoder convolution block
    """

    def __init__(self, scale, **args):
        super(UpConvBlock, self).__init__()
        self.conv_block = ConvBlock(**args)
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.upsample(x)
        return x


class AutoEncoder1D(nn.Module):
    """
    This is the final Encoder-Decoder model with skip connections
    """

    def __init__(self,
                 n_electrodes=30,  # Number of channels
                 n_freqs=16,  # Number of wavelets
                 n_channels_out=21,  # Number of fingers
                 channels=[8, 16, 32, 32],  # Number of features on each encoder layer
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4],
                 dilation=[1, 1, 1]
                 ):

        super(AutoEncoder1D, self).__init__()

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs * n_electrodes
        self.n_channels_out = n_channels_out

        self.model_depth = len(channels) - 1
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)  # Dimensionality reduction

        # Encoder part
        self.downsample_blocks = nn.ModuleList([ConvBlock(channels[i],
                                                          channels[i + 1],
                                                          kernel_sizes[i],
                                                          stride=strides[i],
                                                          dilation=dilation[i]) for i in range(self.model_depth)])

        channels = [ch for ch in channels[:-1]] + channels[-1:]  # channels

        # Decoder part
        self.upsample_blocks = nn.ModuleList([UpConvBlock(scale=strides[i],
                                                          in_channels=channels[i + 1] if i == self.model_depth - 1 else
                                                          channels[i + 1] * 2,
                                                          out_channels=channels[i],
                                                          kernel_size=kernel_sizes[i]) for i in
                                              range(self.model_depth - 1, -1, -1)])

        self.conv1x1_one = nn.Conv1d(channels[0] * 2, self.n_channels_out, kernel_size=1,
                                     padding='same')  # final 1x1 conv

    def forward(self, x):

        batch, elec, n_freq, time = x.shape
        x = x.reshape(batch, -1, time)  # flatten the input
        x = self.spatial_reduce(x)

        skip_connection = []

        for i in range(self.model_depth):
            skip_connection.append(x)
            x = self.downsample_blocks[i](x)

        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)
            x = torch.cat((x, skip_connection[-1 - i]),  # skip connections
                          dim=1)

        x = self.conv1x1_one(x)

        return x
