import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from game import GameState, RandomPlayer,Player, PLAYER1


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, opponent: Player, dimension: int = 9, player: int = PLAYER1, toggle_players: bool = True):
        super().__init__()
        self.dimension = dimension
        self.action_space = spaces.Discrete(dimension)
        # 0 for empty cell, 1 for Player1, 2 for Player2
        self.observation_space = spaces.Box(low=0, high=2, shape=(dimension,), dtype=np.int32)
        self.opponent = opponent
        self.player = player
        self.toggle_players = toggle_players
        self.state = GameState(opponent,self.player, self.toggle_players)

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
        observation, reward, terminated = self.state.make_move(action)
        truncated = False #TODO: Might reconsider
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
        pass

    def close(self):
        pass

if __name__ == "__main__":
    # Check the sanity of the environment
    opponent = RandomPlayer()
    env = CustomEnv(opponent)
    check_env(env)
    env.reset()

    import random

    n_steps = 20
    for step in range(n_steps):
        print(f"Step {step + 1}")
        obs, reward, terminated, truncated, info = env.step(random.randint(0,8))
        done = terminated or truncated
        print(obs[:3])
        print(obs[3:6]) 
        print(obs[6:])
        print("obs=", obs, "reward=", reward, "done=", done)
        if done:
            print("Goal reached!", "reward=", reward)
            break