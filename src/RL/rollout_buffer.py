from typing import Sequence
from collections import deque, namedtuple
import numpy as np

Rollout = namedtuple("Rollout", field_names=["states", "actions", "rewards"])
class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, rollout: Rollout):
        self.buffer.append(rollout)

    def sample(self, idx: int) -> Sequence[np.array]:
        states, actions, rewards = self.buffer[idx]
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        return states, actions, rewards

    def __len__(self) -> int:
        return len(self.buffer)