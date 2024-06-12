import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class EcogFingerflexDataset(Dataset):
    """
    The class that defines the sampling unit
    """

    def __init__(self, path_to_ecog_data: str,
                 path_to_fingerflex_data: str, sample_len: int, train=False):
        """
        paths should point to .npy files
        """
        self.ecog_data, self.fingerflex_data = np.load(path_to_ecog_data).astype('float32'), \
            np.load(path_to_fingerflex_data).astype('float32')

        self.duration = self.ecog_data.shape[2]
        self.sample_len = sample_len  # sample size
        self.stride = 1  # stride between samples
        self.ds_len = (self.duration - self.sample_len) // self.stride
        self.train = train

        print("Duration: ", self.duration, "Ds_len:", self.ds_len)

    def __len__(self):
        return self.ds_len

    def __getitem__(self, index):
        sample_start = index * self.stride
        sample_end = sample_start + self.sample_len

        ecog_sample = self.ecog_data[..., sample_start:sample_end]  # x

        fingerflex_sample = self.fingerflex_data[..., sample_start:sample_end]  # y

        return ecog_sample, fingerflex_sample


class EcogFingerflexDatamodule(pl.LightningDataModule):
    """
    A class that encapsulates different datasets (for training and validation) and their dataloaders
    """
    DATA_PATH_FORMAT = "{data_dir}/{phase}/{data_type}{add_name}.npy"

    def __init__(
            self,
            sample_len: int,
            data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed"),
            batch_size=128,
            add_name=""
    ):
        super().__init__()
        self.data_dir = data_dir  # Path to data folder
        self.sample_len = sample_len  # Sample size
        self.batch_size = batch_size  # Dataloader batch size
        self.add_name = add_name  # dataset name
        self.train, self.val, self.test = None, None, None

    @property
    def ecog_data_val_path(self):
        return self.DATA_PATH_FORMAT.format(
            data_dir=self.data_dir,
            phase="val",
            data_type="ecog_data",
            add_name=self.add_name
        )

    @property
    def fingerflex_data_val_path(self):
        return self.DATA_PATH_FORMAT.format(
            data_dir=self.data_dir,
            phase="val",
            data_type="fingerflex_data",
            add_name=self.add_name
        )

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            self.train = EcogFingerflexDataset(
                f"{self.data_dir}/train/ecog_data{self.add_name}.npy",
                f"{self.data_dir}/train/fingerflex_data{self.add_name}.npy",
                self.sample_len, train=True
            )

            if os.path.exists(self.ecog_data_val_path) and os.path.exists(self.fingerflex_data_val_path):
                self.val = EcogFingerflexDataset(
                    self.ecog_data_val_path,
                    self.fingerflex_data_val_path,
                    self.sample_len
                )

        if stage is None or stage == "test":
            self.test = EcogFingerflexDataset(
                f"{self.data_dir}/test/ecog_data{self.add_name}.npy",
                f"{self.data_dir}/test/fingerflex_data{self.add_name}.npy",
                self.sample_len
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
