import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.tools as tls
import scipy.interpolate
import scipy.io
import scipy.ndimage
import torch
import torch.nn as nn
import wandb
from pytorch_lightning.callbacks import Callback


def load_data(ecog_data_path, fingerflex_data_path):
    ecog_data = np.load(ecog_data_path)
    fingerflex_data = np.load(fingerflex_data_path)
    return ecog_data, fingerflex_data


def correlation_metric(x, y):
    """
     Cosine distance calculation metric
    """
    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)

    cos_sim = torch.mean(cos_metric(x, y))

    return cos_sim


def corr_metric(x, y):
    """
    Pearson correlation calculation metric between univariate vectors
    """
    assert x.shape == y.shape
    r = np.corrcoef(x, y)[0, 1]
    return r


class ValidationCallback(Callback):
    """
    Callback calculating the correlation at the end of each validation epoch on the whole dataset
     and its logging (with visualization) in wandb. In addition, it performs prediction smoothing with Gaussian function
    """
    DOWNSAMPLE_FS = 100  # Desired sampling rate

    def __init__(self, val_x, val_y, fg_num):
        super().__init__()
        self.val_x = val_x.T
        self.val_y = val_y.T
        self.fg_num = fg_num

    def on_validation_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            SIZE = 64
            # SIZE = 256
            bound = self.val_x.shape[0] // SIZE * SIZE

            X_test = self.val_x[:bound]
            y_test = self.val_y[:bound]
            x_batch = torch.from_numpy(X_test).float().to("cuda:3")

            x_batch = x_batch.T

            x_batch = torch.unsqueeze(x_batch, 0)

            y_hat = pl_module.model(x_batch)[0]  # Running data through the model
            y_hat = y_hat.cpu().detach().numpy()
            STRIDE = 1  # It is possible to validate with stride
            y_prediction = y_hat.T[::int(STRIDE * (self.DOWNSAMPLE_FS / 100)), :]
            y_prediction = scipy.ndimage.gaussian_filter1d(y_prediction.T,
                                                           sigma=6).T  # Prediction smoothing with Gaussian function

            y_test = y_test[::int(STRIDE * (self.DOWNSAMPLE_FS / 100)), :]

            h, w = self.fg_num // 2, self.fg_num - self.fg_num // 2
            fig, ax = plt.subplots(h, w, figsize=(h * 5, w * 6), sharex=True,
                                   sharey=True)  # Making pair plots of true motion and prediction
            corrs = []

            for roi in range(self.fg_num):
                y_hat = y_prediction[:, roi]
                y_test_roi = y_test[:, roi]
                corr_tmp = corr_metric(y_hat, y_test_roi)  # Correlation сalculation
                corrs.append(corr_tmp)
                axi = ax.flat[roi]
                axi.plot(y_hat, label='prediction')
                axi.plot(y_test_roi, label='true')

                axi.set_title("RoI {}_corr {:.2f}".format(roi, corr_tmp))

            corr_mean = np.mean(corrs)
            pl_module.log("corr_mean_val", corr_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # wandb.log({"corr_mean_val" : corr_mean })
            wandb.log({f"plots": fig})  # Logging charts


class TestCallback:
    """
    Callback, which calculates the correlation on the whole dataset and visualizes it in case of testing.
    In addition, it also produces exactly the same prediction smoothing with the Gaussian function
    """
    DOWNSAMPLE_FS = 100  # Desired sampling rate

    def __init__(self, val_x, val_y, fg_num):
        super().__init__()
        self.val_x = val_x.T
        self.val_y = val_y.T
        self.fg_num = fg_num

    def test(self, pl_module):
        with torch.no_grad():
            SIZE = 64
            bound = self.val_x.shape[0] // SIZE * SIZE

            X_test = self.val_x[:bound]
            y_test = self.val_y[:bound]
            x_batch = torch.from_numpy(X_test).float()  # .to("cuda:3")

            x_batch = x_batch.T

            x_batch = torch.unsqueeze(x_batch, 0)

            y_hat = pl_module.model(x_batch)[0]
            y_hat = y_hat.cpu().detach().numpy()
            STRIDE = 1
            y_prediction = y_hat.T[::int(STRIDE * (self.DOWNSAMPLE_FS / 100)), :]
            y_prediction = scipy.ndimage.gaussian_filter1d(y_prediction.T, sigma=1).T

            y_test = y_test[::int(STRIDE * (self.DOWNSAMPLE_FS / 100)), :]

            np.save(f"{pathlib.Path().resolve()}/res_npy/prediction2.npy", y_prediction)
            np.save(f"{pathlib.Path().resolve()}/res_npy/true2.npy", y_test)

            h, w = self.fg_num // 2, self.fg_num - self.fg_num // 2
            fig, ax = plt.subplots(h, w, figsize=(h * 35, w * 6), sharex=True, sharey=True)
            corrs = []

            for roi in range(self.fg_num):
                y_hat = y_prediction[:, roi]
                y_test_roi = y_test[:, roi]
                corr_tmp = corr_metric(y_hat, y_test_roi)
                corrs.append(corr_tmp)
                axi = ax.flat[roi]
                axi.plot(y_hat, label='prediction')
                axi.plot(y_test_roi, label='true')

                axi.set_title("RoI {}_corr {:.2f}".format(roi, corr_tmp))

            corr_mean = np.mean(corrs)

            plotly_fig = tls.mpl_to_plotly(fig)  # Converting matplotlib image to plotly
            print(corr_mean)
            plotly_fig.write_html("res.html")  # Writing the interactive visualization to an html file
