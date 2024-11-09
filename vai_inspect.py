from src.raw_models import AutoEncoder1D 
import torch
import argparse
import sys 

# Only works with Vitis AI
from pytorch_nndct.apis import Inspector

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect the model using Vitis AI')
    parser.add_argument(
        '--model', '-m', type=str, default="vai_models\\trained\\model-epoch=16-corr_mean_val=0.6680787205696106.pth",
        help='Path to the model'
    )
    parser.add_argument(
        '--target', '-t', type=str, default="DPUCZDX8G_ISA1_B4096",
        help='Target platform (DPU)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default="vai_inspect_results.txt",
        help='Output file'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    model_to_test = args.model
    hp_autoencoder = dict(
        channels=[32, 32, 64, 64, 128, 128],
        kernel_sizes=[7, 7, 5, 5, 5],
        strides=[2, 2, 2, 2, 2],
        dilation=[1, 1, 1, 1, 1],
        n_electrodes=62,
        n_freqs=40,
        n_channels_out=5
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder1D(**hp_autoencoder)
    model.load_state_dict(torch.load(model_to_test, map_location=device ,weights_only=False))

    print("Model loaded successfully")
    target = args.target
    inspector = Inspector(target)
    
    input = torch.randn(4, 62, 40, 256)
    inspector.inspect(model, input, device=device)

if __name__ == "__main__":
    sys.exit(main())