import argparse
import sys

import torch
import wandb
import pathlib
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_model_summary import summary

from src.dataset import EcogFingerflexDatamodule
from src.models import AutoEncoder1D, BaseEcogFingerflexModel
from src.tools import ValidationCallback


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument(
        '--channels_num', type=int, default=62,
        help='Number of channels in the ECoG data'
    )
    parser.add_argument(
        '--wavelet_num', type=int, default=40,
        help='Number of wavelets in the indicated frequency range, with which the convolution is performed'
    )
    parser.add_argument(
        '--finger_num', type=int, default=5,
        help='Number of fingers'
    )
    parser.add_argument(
        '--sample_len', type=int, default=256,
        help='Window size'
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    channels_num = args.channels_num
    wavelet_num = args.wavelet_num
    finger_num = args.finger_num
    sample_len = args.sample_len
    device = args.device
    
    # Torch optimization for RTX 4070 SUPER
    torch.set_float32_matmul_precision('medium')

    # Model hyperparameters
    hp_autoencoder = dict(
        channels=[32, 32, 64, 64, 128, 128],
        kernel_sizes=[7, 7, 5, 5, 5],
        strides=[2, 2, 2, 2, 2],
        dilation=[1, 1, 1, 1, 1],
        n_electrodes=channels_num,
        n_freqs=wavelet_num,
        n_channels_out=finger_num
    )

    model = AutoEncoder1D(**hp_autoencoder).to(device)
    lightning_wrapper = BaseEcogFingerflexModel(model)

    dm = EcogFingerflexDatamodule(
        sample_len=sample_len,
        add_name=""
    )
    summary(
        model,
        torch.zeros(4, channels_num, wavelet_num, sample_len).to(device),
        show_input=False
    )
    
    SAVE_PATH = f"{pathlib.Path().resolve()}/data/preprocessed"

    def load_data(ecog_data_path, fingerflex_data_path):
        ecog_data = np.load(ecog_data_path)
        fingerflex_data = np.load(fingerflex_data_path)
        return ecog_data, fingerflex_data

    # Validation data
    ecog_data_val, fingerflex_data_val = load_data(f"{SAVE_PATH}/val/ecog_data.npy", f"{SAVE_PATH}/val/fingerflex_data.npy")

    # wandb.init(project="BCI_comp")
    # wandb_logger = WandbLogger()
    csv_logger = CSVLogger("logs", name="ecog_fingerflex")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="corr_mean_val",
        # monitor="corr_mean",
        mode="max",
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{corr_mean_val}",
    )

    trainer = Trainer(
        max_epochs=20,
        logger=csv_logger,
        callbacks=[
            ValidationCallback(
                ecog_data_val,
                fingerflex_data_val,
                finger_num
            ),
            checkpoint_callback,
        ]
    )
    trainer.fit(lightning_wrapper, dm)
    # wandb.finish()


if __name__ == "__main__":
    sys.exit(main())
