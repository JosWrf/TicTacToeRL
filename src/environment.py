import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from game import GameState


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, dimension: int = 9):
        super().__init__()
        self.dimension = dimension
        self.action_space = spaces.Discrete(dimension)
        # 0 for empty cell, 1 for Player1, 2 for Player2
        self.observation_space = spaces.Box(low=0, high=2, shape=(dimension,), dtype=np.int32)
        self.state = GameState()

    def step(self, action):
        """Called to take an action with the environment.

        Args:
            action (_type_): _description_

        Returns:
            Tuple[Any, float, bool, bool, Dict[]]: Next Observation, Immediate Reward,
            Terminal state, Max timesteps reached, additional info
        """
        if  0 > action  or action > 8:
            raise ValueError(f"Received invalid action={action}!")
        #TODO: What to do when the action is not valid since an entry is != 0 ?! 
        observation = ...
        reward = ...
        terminated = ...
        truncated = ...
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed:int|None=None, options=None):
        """Called at the beginning of an episode.

        Args:
            seed (int|None, optional): Seed. Defaults to 17.
            options (None): Optional options dict. Defaults to None.

        Returns:
            np.ndarray: Observation
        """
        super().reset(seed=seed, options=options)
        self.state.reset_state()
        observation = self.state.get_board_state()
        info = {}
        return observation, info

    def render(self):
        """(Optional) Allows to visualize the agent in action.
        """
        ...

    def close(self):
        pass

if __name__ == "__main__":
    # Check the sanity of the environment
    env = CustomEnv()
    print(env.action_space.sample())
    print(env.observation_space.sample())
    #check_env(env)