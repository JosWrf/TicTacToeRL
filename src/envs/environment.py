import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from envs.game import GameState, RandomPlayer, Player, PLAYER1


class TicTacToe(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["console"], "render_fps": 30}

    def __init__(
        self,
        opponent: Player,
        dimension: int = 9,
        player: int = PLAYER1,
        toggle_roles: bool = True,
        render_mode: str = "console",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dimension = dimension
        self.action_space = spaces.Discrete(dimension)
        # 0 for empty cell, 1 for Player1, 2 for Player2
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(dimension,), dtype=np.int32
        )
        self.opponent = opponent
        self.player = player
        self.toggle_roles = toggle_roles
        self.state = GameState(opponent, self.player, toggle_roles)

    def step(self, action):
        """Called to take an action with the environment.

        Args:
            action (int): Action in [0,8] encoding the position on the grid.

        Returns:
            Tuple[Any, float, bool, bool, Dict[]]: Next Observation, Immediate Reward,
            Terminal state, Max timesteps reached, additional info
        """
        if 0 > action or action > 8:
            raise ValueError(f"Received invalid action={action}!")
        observation, reward, terminated = self.state.make_move(action)
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options=None):
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
        """(Optional) Allows to visualize the agent in action."""
        if self.render_mode == "console":
            state = self.state.get_board_state()
            print(state[:3])
            print(state[3:6])
            print(state[6:])

    def close(self):
        pass  # no ressources to clean up after

if __name__ == "__main__":
    # Check the sanity of the environment
    opponent = RandomPlayer()
    env = TicTacToe(opponent)
    check_env(env)