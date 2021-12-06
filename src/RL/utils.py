import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rollout_buffer import RolloutBuffer, Rollout

def get_cumulative_rewards(rewards, gamma=0.99):
    G = [rewards[-1]]
    for r in reversed(rewards[:-1]):
        G.insert(0, r + gamma * G[0])
    return G


def to_one_hot(y, n_dims=None):
    """Takes an integer vector and converts it to 1-hot matrix."""
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot

def get_action(env, network: nn.Module, state: np.array) -> int:
    state = torch.tensor(state[None], dtype=torch.float32)
    logits = network(state).detach()
    probas = F.softmax(logits, -1).cpu().numpy()[0]
    action = np.random.choice(len(probas), p=probas)
    return int(action)

def generate_session(
    env, network: nn.Module, t_max: int = 1000, rollout_buffer: Optional[RolloutBuffer] = None
) -> Tuple[float, int]:
    total_reward = 0
    states, actions, rewards = [], [], []
    state = env.reset()

    for t in range(t_max):
        action = get_action(env, network, state=state)
        next_state, reward, done, _ = env.step(action)

        # record session history to train later
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        total_reward += reward
        state = next_state
        if done:
            break
    if rollout_buffer is not None:
        rollout_buffer.append(Rollout(states, actions, rewards))

    return total_reward, t


def generate_sessions(
    env,
    network: nn.Module,
    t_max: int = 1000,
    rollout_buffer: Optional[RolloutBuffer] = None,
    num_sessions: int = 100,
) -> Tuple[float, int]:
    sessions_reward, sessions_steps = 0, 0
    for i_episone in range(num_sessions):
        r, t = generate_session(
            env=env, network=network, t_max=t_max, rollout_buffer=rollout_buffer
        )
        sessions_reward += r
        sessions_steps += t
    return sessions_reward, sessions_steps