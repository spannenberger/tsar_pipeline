from catalyst.dl import registry, Callback, CallbackOrder, State
import torch
from pathlib import Path
from catalyst.utils.tracing import (
    save_traced_model,
    trace_model_from_checkpoint,
)
@registry.Callback
class TorchscriptSaveCallback(Callback):

    def __init__(self,
                 method_name = "forward",
                 checkpoint_names = ["best"],
                 mode = "eval",
                 device = "cuda:0",
                 requires_grad = False,
                 out_dir = "./tochsript"):

        self.method_name = method_name
        self.method_name = method_name
        self.checkpoint_names = checkpoint_names
        self.mode = mode
        self.device = device
        self.requires_grad = requires_grad
        self.out_dir = Path(out_dir)
        super().__init__(CallbackOrder.Internal)

    def on_experiment_end(self, state: State):
        for checkpoint in self.checkpoint_names:
            traced_model = trace_model_from_checkpoint(
                logdir = state.logdir,
                method_name = self.method_name,
                checkpoint_name = checkpoint,
                mode = self.mode,
                requires_grad = self.requires_grad,
                device = self.device
            )
            save_traced_model(
                model = traced_model,
                logdir = state.logdir,
                method_name = self.method_name,
                checkpoint_name = checkpoint,
                mode = self.mode,
                requires_grad = self.requires_grad,
                out_dir = self.out_dir
            )
