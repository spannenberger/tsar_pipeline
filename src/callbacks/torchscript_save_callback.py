from catalyst.dl import registry, Callback, CallbackOrder, State
import torch
from pathlib import Path
from catalyst.utils.tracing import (
    save_traced_model,
    trace_model_from_checkpoint,
)
@registry.Callback
class TorchscriptSaveCallback(Callback):

    def __init__(self,out_dir,checkpoint_names):
        self.out_dir = Path(out_dir)
        self.checkpoint_names = checkpoint_names
        super().__init__(CallbackOrder.Internal)

    def on_experiment_end(self, state: State):
        for checkpoint in self.checkpoint_names:
            path = state.logdir / "checkpoints" / (checkpoint + ".pth")
            state.model.load_state_dict(torch.load(path)['model_state_dict'])
            state.model.eval()

            scripted = torch.jit.script(state.model,torch.rand(1,3, 512,512))
            path = state.logdir/self.out_dir/(checkpoint+".pt")
            path.parent.mkdir(parents = True,exist_ok = True)
            torch.jit.save(scripted, str(path))
