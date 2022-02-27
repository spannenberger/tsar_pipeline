from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path
from tqdm import tqdm
import warnings
import torch


@Registry
class OnnxSaveCallback(Callback):

    def __init__(self, out_dir, checkpoint_names):
        super().__init__(CallbackOrder.External)
        self.out_dir = Path(out_dir)
        self.checkpoint_names = checkpoint_names

    def on_experiment_end(self, state: IRunner):
        for checkpoint in tqdm(self.checkpoint_names, desc="Converting to onnx"):
            try:
                model_path = Path(state.logdir).absolute() / "checkpoints" / (checkpoint + ".pth")
                state.model.load_state_dict(torch.load(model_path)['model_state_dict'])
            except FileNotFoundError:
                warnings.warn(
                    "File for onnx convertation is not found"
                )
                continue
            state.model.eval()
            state.engine.sync_device(state.model)
            x = torch.randn(1, 3, 224, 224, requires_grad=True, device=state.engine.device)
            path = Path(state.logdir).absolute()/self.out_dir/(checkpoint+'.onnx')
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.onnx.export(state.model,               # model being run
                              # model input (or a tuple for multiple inputs)
                              x,
                              str(path),   # where to save the model (can be a file or file-like object)
                              export_params=True,        # store the trained parameter weights inside the model file
                              opset_version=11,          # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input'],   # the model's input names
                              output_names=['output'],  # the model's output names
                              dynamic_axes={'input': {0: 'batch_size'},    # variable lenght axes
                                            'output': {0: 'batch_size'}})
            except RuntimeError:
                warnings.warn(
                    "Can't convert this model to onnx format"
                )
