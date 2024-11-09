from src.dataset import EcogFingerflexDatamodule
from src.raw_models import AutoEncoder1D 
import torch
import argparse
import sys 
from tqdm import tqdm

# Only works with Vitis AI
from pytorch_nndct.apis import Inspector
from pytorch_nndct.apis import torch_quantizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument('--quant_mode', 
        default='calib', 
        choices=['float', 'calib', 'test'], 
        help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
    parser.add_argument(
        '--config_file',
        default=None,
        help='quantization configuration file')
    return parser.parse_args()

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions
    for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def forward_loop(model, val_loader):
  model.eval()
  model = model.to(device)
  for iteraction, (images, _) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
    images = images.to(device)
    outputs = model(images)


# Evalution function should be called in quantization test stage. 
def evaluate(model, val_loader, loss_fn):
  model.eval()
  model = model.to(device)
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  total = 0
  Loss = 0
  for iteraction, (images, labels) in tqdm(
      enumerate(val_loader), total=len(val_loader)):
    images = images.to(device)
    labels = labels.to(device)
    #pdb.set_trace()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    Loss += loss.item()
    total += images.size(0)
    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))
  return top1.avg, top5.avg, Loss / total

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

    model = AutoEncoder1D(**hp_autoencoder).to(device)
    model.load_state_dict(torch.load(model_to_test, map_location=device ,weights_only=False))

    print("Model loaded successfully")
    target = args.target
    
    # print("Inspecting model...")
    # inspector = Inspector(target)
    # inspector.inspect(model, input, device=device)
    
    print("Quantizing model...")
    config_file = args.config_file
    quant_mode = args.quant_mode
    batch_size = 128
    sample_len = 256
    if quant_mode == 'test':
        batch_size = 1
    #     sample_len = 1
    deploy = True
    input = torch.randn(4, 62, 40, 256)
    quantizer = torch_quantizer(quant_mode=quant_mode, quant_config_file=config_file, target=target, input_args=(input), device=device, module=model)

    dm = EcogFingerflexDatamodule(
        sample_len=sample_len,
        batch_size=batch_size,
        add_name=""
    )
    dm.setup()
    val_loader = dm.val_dataloader()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    quant_model = quantizer.quant_model

    if quant_mode == 'calib':
        # This function call is to do forward loop for model to be quantized.
        # Quantization calibration will be done after it.
        forward_loop(quant_model, val_loader)
        # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
        quantizer.export_quant_config()
    else:
        forward_loop(quant_model, val_loader)
    
    
    # quantizer.export_torch_script()
    print("Model quantized successfully")
    print("Exporting model...")
    quantizer.export_xmodel(output_dir='vai_models/quantized')
    print("Model exported successfully")
    
    
    
if __name__ == "__main__":
    sys.exit(main())