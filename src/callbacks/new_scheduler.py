from catalyst.dl import registry, Callback, CallbackOrder, State
from torch import optim, nn
from torchtools.lr_scheduler import DelayerScheduler

@registry.Scheduler
class CustomScheduler(DelayerScheduler):
    def __init__(self, lr=None, delay_epochs=None, total_epochs=None, optimizer=None, eta_min=None):
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs - delay_epochs, eta_min)
        super().__init__(optimizer, delay_epochs, base_scheduler)