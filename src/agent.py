from copy import deepcopy
import random
from typing import Any, Dict
import gymnasium
import os
from collections import deque
from envs.game import TIE, RLAgent
from stable_baselines3.common.callbacks import BaseCallback

class SelfplayCallback(BaseCallback):
    def __init__(
        self,
        verbose: int = 0,
        size: int = 10,
        start_collecting: int = 10000,
        collect_frequ: int = 100,
        change_agents: int = 200,
        change_prob: float = 0.8,
        save_model: bool = False,
        model_name: str = "PPO-TicTacToe",
    ):
        super(SelfplayCallback, self).__init__(verbose)
        self.policies = deque(maxlen=size)
        self.start_collecting = start_collecting
        self.change_agents = change_agents
        self.collect_frequ = collect_frequ
        self.change_prob = change_prob
        self.save_model = save_model
        self.model_name = model_name

    def _on_training_start(self) -> None:
        if self.save_model:
            os.makedirs("./models", exist_ok=True)

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        if (
            len(self.policies) > self.start_collecting 
            and self.num_timesteps % self.change_agents == 0
            and self.change_prob > random.random()
        ):
            new_policy = self.policies[random.randint(0, len(self.policies) - 1)]
            self.training_env.get_attr("state")[0].set_opponent(RLAgent(new_policy))

        return True

    def _on_training_end(self) -> None:
        if self.save_model:
            self.model.save(f"models/{self.model_name}")

    def _on_rollout_end(self) -> None:
        """Buffer the policies of previous agents for self-play."""
        if self.num_timesteps < self.start_collecting:
            return
        self.num_timesteps % (self.start_collecting + self.collect_frequ) == 0
        policy = deepcopy(self.model.policy)
        if len(self.policies) == self.policies.maxlen - 1:
            self.policies.popleft()
        self.policies.append(policy)
    
def evaluate(model: Any, env: gymnasium.Env, episodes:int = 100) -> Dict[str, int]:
    obs = env.reset()
    stats = {"T": 0, "W": 0, "L": 0}
    for i in range(episodes):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                if reward < 0:
                    stats["L"] += 1
                elif reward == TIE:
                    stats["T"] += 1
                else:
                    stats["W"] += 1

    return stats