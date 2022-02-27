from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path
from tqdm import tqdm
import warnings
import torch


@Registry
class TorchscriptSaveCallback(Callback):

    def __init__(self, out_dir, checkpoint_names):
        super().__init__(CallbackOrder.External)
        self.out_dir = Path(out_dir)
        self.checkpoint_names = checkpoint_names

    def on_experiment_end(self, state: IRunner):
        for checkpoint in tqdm(self.checkpoint_names, desc="Converting to torchscript"):
            try:
                model_path = Path(state.logdir).absolute() / "checkpoints" / (checkpoint + ".pth")
                state.model.load_state_dict(torch.load(model_path)['model_state_dict'])
            except FileNotFoundError:
                warnings.warn(
                    "File for torchscript convertation is not found"
                )
                continue
            state.model.eval()
            state.engine.sync_device(state.model)
            x = torch.randn(1, 3, 224, 224, requires_grad=True, device=state.engine.device)

            path = Path(state.logdir).absolute() / self.out_dir / (checkpoint+'.pt')
            path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                with torch.jit.optimized_execution(True):
                    scripted = torch.jit.script(state.model, x)
                    try:
                        torch.jit.save(scripted, str(path))
                    except RuntimeError:
                        warnings.warn(
                            "Can't convert this model to torchscript format"
                        )
