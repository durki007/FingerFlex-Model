from src.models import BaseEcogFingerflexModel 
from src.raw_models import AutoEncoder1D
from src.tools import TestCallback
import torch

model_to_test = "checkpoints\model-epoch=01-corr_mean_val=0.6366250514984131.ckpt"
save_dir ="vai_models/trained/"
save_name = "2d.pth"
save_path = save_dir + save_name


hp_autoencoder = dict(
    channels=[32, 32, 64, 64, 128, 128],
    kernel_sizes=[7, 7, 5, 5, 5],
    strides=[2, 2, 2, 2, 2],
    dilation=[1, 1, 1, 1, 1],
    n_electrodes=62,
    n_freqs=40,
    n_channels_out=5
)

trained_model = BaseEcogFingerflexModel.load_from_checkpoint(
    checkpoint_path=model_to_test,
    model=AutoEncoder1D(**hp_autoencoder))

print("Model loaded successfully")
print("Dropping Lightning wrapper...")

raw_model = trained_model.model
torch.save(raw_model.state_dict(), save_path) 

print(f"Model saved to {save_path}")
