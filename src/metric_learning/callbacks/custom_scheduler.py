from torch import optim
from torchtools.lr_scheduler import DelayerScheduler
from catalyst.registry import Registry


@Registry
class CustomScheduler(DelayerScheduler):
    """
    Псевдо кастомный шедулер
    Спустя указанный delay уменьшает learning rate по косинусу
    """

    def __init__(self, lr=None, delay_epochs=None, total_epochs=None, optimizer=None, eta_min=None):
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs - delay_epochs, eta_min)
        super().__init__(optimizer, delay_epochs, base_scheduler)
