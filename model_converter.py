import argparse
from pathlib import Path
import torch
import yaml
import sys
import os
import onnxruntime as rt


def parse_args():
    parser = argparse.ArgumentParser(description='Convert checkpoints .pth to other format')
    parser.add_argument(
        '--input_dir',
        default='logs/checkpoints',
        help='Input dir for checkpoints .pth'
    )
    parser.add_argument(
        '--model_file',
        default='src/classification/models/Densenet121.py',
        help='File of model'
    )
    parser.add_argument(
        '--model_name',
        default='Densenet121',
        help='Model class name'
    )
    parser.add_argument(
        '--out_dir',
        default=None,
        help='Output dir for checkpoints'
    )
    parser.add_argument(
        '--format',
        default='torchscript',
        help='Format into which we convert'
    )
    parser.add_argument(
        '--logdir',
        default='./logs/',
        help='Custom logdir'
    )
    parser.add_argument(
        '--config_file',
        default='config/classification/multilabel/train_multilabel.yml',
        help='Config file with model args'
    )
    return parser.parse_args()


def get_model_params(config_file: str):
    with open(config_file, encoding='utf8') as f:
        params = yaml.safe_load(f)
        params = params['model']
        del params['_target_']
        return params


def torchscript(model: torch.nn.Module, checkpoint: str, input_dir: Path, output_dir: Path):
    device = 'cpu:0'
    if torch.cuda.is_available():
        device = 'cuda:0'
    model.load_state_dict(torch.load(input_dir/(checkpoint+'.pth'))['model_state_dict'])
    model.eval()
    model.to(device)
    x = torch.rand(1, 3, 512, 512)
    x.to(device)
    scripted = torch.jit.script(model, x)
    output = output_dir / (checkpoint + '.pt')
    torch.jit.save(scripted, str(output))
    loaded = torch.jit.load(str(output))
    loaded.to(device)
    print(loaded(x) == model(x))


def onnx(model: torch.nn.Module, checkpoint: str, input_dir: Path, output_dir: Path):
    device = 'cpu:0'
    if torch.cuda.is_available():
        device = 'cuda:0'
    model.load_state_dict(torch.load(input_dir / (checkpoint + '.pth'))['model_state_dict'])
    model.eval()
    model.to(device)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    x.to(device)
    path = output_dir / (checkpoint+'.onnx')
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      str(path),   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable lenght axes
                                    'output': {0: 'batch_size'}})

    onnx_input = x.cpu().detach().clone().numpy()
    preds_torch = model(x)

    session = rt.InferenceSession(str(path))
    inputs_name = session.get_inputs()[0].name
    outputs_name = session.get_outputs()[0].name
    preds_onx = session.run([outputs_name], {inputs_name: onnx_input})[0]
    print(preds_torch.detach())
    print(torch.Tensor(preds_onx))


converters = {
    'torchscript': torchscript,
    'onnx': onnx,
}

args = parse_args()
input_dir = Path(args.input_dir)
if args.out_dir is None:
    output_dir = Path(args.logdir + args.format)
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

model_params = get_model_params(args.config_file)
chechpoints = ['best', 'best_full']
model_file = args.model_file.split('/')
model_file = '.'.join(model_file)[:-3]
sys.path.append(os.getcwd() + '/' + '/'.join(args.model_file.split('/')[:-2]))
model_module = __import__(model_file, fromlist=[None])

model = getattr(model_module, args.model_name)(**model_params)

for checkpoint in chechpoints:
    converters[args.format](model, checkpoint, input_dir, output_dir)
