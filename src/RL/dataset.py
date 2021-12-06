from .rollout_buffer import RolloutBuffer
from typing import Iterator, Sequence
import numpy as np
from torch.utils.data.dataset import IterableDataset

class RolloutDataset(IterableDataset):
    def __init__(self, buffer: RolloutBuffer):
        self.buffer = buffer

    def __iter__(self) -> Iterator[Sequence[np.array]]:
        for i in range(len(self.buffer)):
            states, actions, rewards = self.buffer.sample(i)
            yield states, actions, rewards
        self.buffer.buffer.clear()

    def __len__(self) -> int:
        return self.buffer.capacity