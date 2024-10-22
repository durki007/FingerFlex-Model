from src.models import BaseEcogFingerflexModel, AutoEncoder1D 
from src.tools import TestCallback

model_to_test = "checkpoints\model-epoch=17-corr_mean_val=0.9735080003738403.ckpt"
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
test_callback = TestCallback(ecog_data_val, fingerflex_data_val, finger_num)
test_callback.test(trained_model) # Testing