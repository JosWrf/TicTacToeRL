from typing import Any, Dict
import optuna
from stable_baselines3 import PPO
from agent import SelfplayCallback, evaluate
from envs.environment import TicTacToe
from stable_baselines3.common.env_util import make_vec_env

from envs.game import RandomPlayer

N_TRIALS = 200
TRAIN_SEED = 17
TEST_SEED = 2

DEFAULT_PARAMS = {"policy": "MlpPolicy", "verbose": 0}


def sample_ppo_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.01, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    learning_rate = trial.suggest_float("l_r", 1e-5, 0.01, log=True)
    n_epochs = trial.suggest_int("epochs", 10, 20)

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "n_epochs": n_epochs,
    }


def objective(trial: optuna.Trial) -> float:
    params = sample_ppo_hyperparams(trial)
    params.update(DEFAULT_PARAMS)

    opponent = RandomPlayer()
    env = TicTacToe(opponent)
    env = make_vec_env(TicTacToe, n_envs=1, env_kwargs=dict(opponent=opponent), seed=TRAIN_SEED)
    params["env"] = env

    model = PPO(**params)

    selfplay = SelfplayCallback()
    time_steps = trial.suggest_int("total_timesteps", 1e5, 1e6)
    learn_params = {"callback": selfplay, "total_timesteps": time_steps}

    nan = False
    try:
        model = model.learn(**learn_params)
    except AssertionError as e:
        print(e)
        nan = True
    if nan:
        return float("nan")

    test_env = make_vec_env(
        TicTacToe, env_kwargs=dict(opponent=RandomPlayer()), seed=TEST_SEED
    )
    stats = evaluate(model, test_env, 1000)
    val = (stats["W"] + stats["T"]) / sum(stats.values()) # Use win ratio against random player as criteria

    try:
        if trial.study.best_value < val:
            trial.set_user_attr("model", model)
    except:
        trial.set_user_attr("model", model)

    return val


study = optuna.create_study(
    study_name="PPO-TicTacToe",
    direction="maximize",
    load_if_exists=True,
)
try:
    study.optimize(objective, n_trials=N_TRIALS)

except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

print("Saving Model ")
trial.user_attrs["model"].save("./model/optModel")