from envs.environment import TicTacToe
from envs.game import TIE, RandomPlayer


if __name__ == "__main__":
    # Check the sanity of the environment
    opponent = RandomPlayer()
    env = TicTacToe(opponent)
    env.reset()

    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(TicTacToe, n_envs=1, env_kwargs=dict(opponent=opponent, toggle_roles=True), seed=2)

    model = DQN("MlpPolicy", env, learning_rate=0.001, verbose=1)
    print(model.policy)
    model = model.learn(500000)
    # Test the trained agent
    # using the vecenv
    obs = env.reset()
    episodes = 100
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