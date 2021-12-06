from catalyst import dl, metrics
from typing import Sequence
import torch
import numpy as np
from .utils import get_cumulative_rewards, to_one_hot
import torch.nn.functional as F

class CustomRunner(dl.Runner):
    def __init__(self, *, gamma: float=0.99, entropy_coef: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma: float = gamma
        self.entropy_coef: float = entropy_coef

    def on_loader_start(self, runner: dl.IRunner):
        super().on_loader_start(runner)
        self.meters = {key: metrics.AdditiveValueMetric(compute_on_call=False) for key in ["loss"]}

    def handle_batch(self, batch: Sequence[np.array]):
        # model train/valid step
        # ATTENTION:
        #   because of different trajectories lens
        #   ONLY batch_size==1 supported
        states, actions, rewards = batch
        states, actions, rewards = states[0], actions[0], rewards[0]
        cumulative_returns = torch.tensor(get_cumulative_rewards(rewards, self.gamma))

        logits = self.model(states)
        probas = F.softmax(logits, -1)
        logprobas = F.log_softmax(logits, -1)
        n_actions = probas.shape[1]
        logprobas_for_actions = torch.sum(logprobas * to_one_hot(actions, n_dims=n_actions), dim=1)

        J_hat = torch.mean(logprobas_for_actions * cumulative_returns)
        entropy_reg = -torch.mean(torch.sum(probas * logprobas, dim=1))
        loss = -J_hat - self.entropy_coef * entropy_reg

        self.batch_metrics.update({"loss": loss})
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner: dl.IRunner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
