import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import VecFrameStack

from BatteryEnv import BatteryEnv
from helpers import evaluate_strategy

def train_and_test_PPOc(
    train, test, simdata, forecast_type, sell_price_ratio=1.0, SEED=2025,
    model_save_path=None, load_existing_model=False, model_load_path=None,
    n_envs=8, n_stack=4,
    learning_rate=3e-4, batch_size=256, gamma=0.99, gae_lambda=0.97, n_steps=336, total_timesteps=300_000
):
    """
    Trains and tests a PPO agent (continuous actions) on the provided train/test splits.
    Only the main RL hyperparameters are tunable.
    """
    # Select forecast column based on forecast_type
    forecast_col = {
        "perfect": "perfect",
        "AR": "AR",
        "LEAR": "LEAR"
    }[forecast_type]

    # Create vectorized and stacked training environment (continuous actions)
    def make_env():
        return BatteryEnv(
            train["prices"], train[forecast_col], train["generation"], train["demand"], train["weather"],
            battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
            discrete_actions=False, sell_price_ratio=sell_price_ratio
        )
    vec_train_env = make_vec_env(make_env, n_envs=n_envs, seed=SEED)
    stacked_train_env = VecFrameStack(vec_train_env, n_stack=n_stack)

    if load_existing_model and model_load_path is not None:
        print(f"Loading existing PPO continuous model from {model_load_path}")
        model = PPO.load(model_load_path)
    else:
        # PPO hyperparameters
        ppo_params = dict(
            policy="MlpPolicy",
            env=stacked_train_env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.005,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=SEED,
        )
        print(f"Training PPO (continuous) using {forecast_type} forecasts and sell_price_ratio={sell_price_ratio}")
        model = PPO(**ppo_params)
        model.learn(total_timesteps=total_timesteps, callback=ProgressBarCallback())
        if model_save_path is not None:
            model.save(model_save_path)

    print(f"Testing PPO (continuous) using {forecast_type} forecasts and sell price ratio: {sell_price_ratio}")
    # Create stacked testing environment
    def make_test_env():
        return BatteryEnv(
            test["prices"], test[forecast_col], test["generation"], test["demand"], test["weather"],
            battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
            discrete_actions=False, sell_price_ratio=sell_price_ratio
        )
    vec_test_env = make_vec_env(make_test_env, n_envs=1, seed=SEED)
    stacked_test_env = VecFrameStack(vec_test_env, n_stack=n_stack)

    obs = stacked_test_env.reset()
    done = False
    total_reward_rl = 0
    total_profit_rl = 0

    actions_rl, soc_list_rl, profit_list_rl = [], [], []
    grid_import_list, grid_export_list = [], []

    while not done:
        action, _states = model.predict(obs)
        actions_rl.append(action)
        obs, reward, done, info = stacked_test_env.step(action)
        total_reward_rl += reward[0]
        total_profit_rl += info[0].get("profit", 0)
        soc_list_rl.append(info[0].get("delta_soc", 0))  # Make sure your env puts soc in info
        profit_list_rl.append(total_profit_rl)
        grid_import_list.append(info[0].get("grid_energy_bought", 0))
        grid_export_list.append(info[0].get("grid_energy_sold", 0))

    df_rl_result_test = pd.DataFrame({
        "action": actions_rl,
        "soc_kWh": soc_list_rl,
        "cumulative_profit": profit_list_rl,
        "grid_import_kWh": grid_import_list,
        "grid_export_kWh": grid_export_list,
    })

    df_rl_result_test.index = test["prices"].index[:len(df_rl_result_test)]

    df_rl_result_2024 = df_rl_result_test.loc["2024-01-01 00:00:00":"2024-12-31 23:00:00"]

    # Evaluate RL results
    no_batt_curve_rl, batt_profit_curve_rl, savings_rl, savings_pct = evaluate_strategy(
        df_full=simdata,
        df_battery_result=df_rl_result_2024,
        start="2024-01-01",
        end="2024-12-31 23:00"
    )

    return df_rl_result_2024, no_batt_curve_rl, batt_profit_curve_rl, savings_rl, savings_pct