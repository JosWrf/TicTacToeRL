from copy import deepcopy
import random
from typing import Any
import gymnasium
import os
from collections import deque
from envs.environment import TicTacToe
from envs.game import TIE, RLAgent, RandomPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class SelfplayCallback(BaseCallback):
    def __init__(
        self,
        verbose: int = 0,
        eval_steps: int = 1000,
        eval_episodes: int = 10,
        size: int = 10,
        start_collecting: int = 50000,
        collect_frequ: int = 100,
        change_agents: int = 200,
        change_prob: float = 0.8,
        model_name: str = "PPO-TicTacToe",
        skip_eval: bool = True
    ):
        super(SelfplayCallback, self).__init__(verbose)
        self.policies = deque(maxlen=size)
        self.start_collecting = start_collecting
        self.change_agents = change_agents
        self.collect_frequ = collect_frequ
        self.change_prob = change_prob
        self.eval_steps = eval_steps
        self.eval_episodes = eval_episodes
        self.model_name = model_name
        self.skip_eval = skip_eval

    def _on_training_start(self) -> None:
        os.makedirs("./models", exist_ok=True)

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        if not self.skip_eval and self.num_timesteps > 0 and self.num_timesteps % self.eval_steps == 0:
            test_env = make_vec_env(TicTacToe, env_kwargs=dict(opponent=RandomPlayer()), seed=17)
            evaluate(self.model, test_env, self.eval_episodes)

        if (
            len(self.policies) > self.start_collecting 
            and self.num_timesteps % self.change_agents == 0
            and self.change_prob > random.random()
        ):
            new_policy = self.policies[random.randint(0, len(self.policies) - 1)]
            self.training_env.get_attr("state")[0].set_opponent(RLAgent(new_policy))

        return True

    def _on_training_end(self) -> None:
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

def learn_policy() -> Any:
    opponent = RandomPlayer()
    env = TicTacToe(opponent)
    env.reset()
    env = make_vec_env(TicTacToe, n_envs=1, env_kwargs=dict(opponent=opponent), seed=2)

    selfplay = SelfplayCallback()

    model = PPO("MlpPolicy", env, learning_rate=0.001, verbose=1)
    model = model.learn(500000, callback=selfplay)
    return model
    
def evaluate(model: Any, env: gymnasium.Env, episodes:int = 100) -> None:
    obs = env.reset()
    stats = {"ties": 0, "wins": 0, "losses": 0}
    for i in range(episodes):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                if reward < 0:
                    stats["losses"] += 1
                elif reward == TIE:
                    stats["ties"] += 1
                else:
                    stats["wins"] += 1

    print(stats)
    
def load_policy(path: str):
    model = PPO.load(path)
    test_env = make_vec_env(TicTacToe, env_kwargs=dict(opponent=RandomPlayer()), seed=17)
    evaluate(model, test_env, 10000)
    

if __name__ == "__main__":
    #model = learn_policy()
    #test_env = make_vec_env(TicTacToe, env_kwargs=dict(opponent=RandomPlayer()), seed=17)
    #evaluate(model, test_env)
    load_policy("./models/PPO-TicTacToe.zip")
