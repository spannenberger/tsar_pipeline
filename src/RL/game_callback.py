from .rollout_buffer import RolloutBuffer
from catalyst import dl
from .utils import generate_sessions
import gym

class GameCallback(dl.Callback):
    def __init__(
        self,
        *,
        env_name,
        rollout_buffer: RolloutBuffer,
        num_train_sessions: int = int(1e2),
        num_valid_sessions: int = int(1e2),
    ):
        super().__init__(order=0)
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.rollout_buffer = rollout_buffer
        self.num_train_sessions = num_train_sessions
        self.num_valid_sessions = num_valid_sessions

    def on_epoch_start(self, runner: dl.IRunner):
        self.actor = runner.model

        self.actor.eval()
        train_rewards, train_steps = generate_sessions(
            env=self.env,
            network=self.actor,
            rollout_buffer=self.rollout_buffer,
            num_sessions=self.num_train_sessions,
        )
        train_rewards /= float(self.num_train_sessions)
        train_steps /= float(self.num_train_sessions)
        runner.epoch_metrics["_epoch_"]["t_reward"] = train_rewards
        runner.epoch_metrics["_epoch_"]["t_steps"] = train_steps
        self.actor.train()

    def on_epoch_end(self, runner: dl.IRunner):
        self.actor.eval()
        valid_rewards, valid_steps = generate_sessions(
            env=self.env, network=self.actor, num_sessions=self.num_valid_sessions
        )
        self.actor.train()

        valid_rewards /= float(self.num_valid_sessions)
        valid_steps /= float(self.num_valid_sessions)
        runner.epoch_metrics["_epoch_"]["v_reward"] = valid_rewards
        runner.epoch_metrics["_epoch_"]["v_steps"] = valid_steps

