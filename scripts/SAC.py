import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor   
import os

from BatteryEnv import BatteryEnv
from helpers import evaluate_strategy

def train_and_test_SAC(
    train, test, simdata, forecast_type, sell_price_ratio=1.0, SEED=2025,
    model_save_path=None, load_existing_model=False, model_load_path=None,
    n_envs=8, n_stack=4,
    learning_rate=3e-4, batch_size=256, gamma=0.99, tau=0.005, ent_coef="auto", total_timesteps=300_000, 
    monitor = False, evaluation = True
):
    """
    Trains and tests a SAC agent (continuous actions) on the provided train/test splits.
    Main RL hyperparameters are tunable.
    """
    # Select forecast column based on forecast_type
    forecast_col = {
        "perfect": "perfect",
        "AR": "AR",
        "LEAR": "LEAR"
    }[forecast_type]

    if monitor:
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        monitor_filename = f"{log_dir}/SAC_{forecast_type}_monitor.csv"

        def make_monitored_env():
            env = BatteryEnv(
                train["prices"], train[forecast_col], train["generation"], train["demand"], train["weather"],
                battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
                discrete_actions=False, sell_price_ratio=sell_price_ratio
            )
            return Monitor(env, filename=monitor_filename)
        vec_train_env = make_vec_env(lambda: make_monitored_env(), n_envs=n_envs, seed=SEED)
    else:
        def make_env():
            return BatteryEnv(
                train["prices"], train[forecast_col], train["generation"], train["demand"], train["weather"],
                battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
                discrete_actions=False, sell_price_ratio=sell_price_ratio
            )
        vec_train_env = make_vec_env(make_env, n_envs=n_envs, seed=SEED)

    stacked_train_env = VecFrameStack(vec_train_env, n_stack=n_stack)
    
    if n_stack > 1:
        stacked_train_env = VecFrameStack(vec_train_env, n_stack=n_stack)
    else:
        stacked_train_env = vec_train_env

    if load_existing_model and model_load_path is not None:
        print(f"Loading existing SAC model from {model_load_path}")
        model = SAC.load(model_load_path)
    else:
        # SAC hyperparameters
        sac_params = dict(
            policy="MlpPolicy",
            env=stacked_train_env,
            verbose=0,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef,
            buffer_size=100_000,
            seed=SEED,
        )
        print(f"Training SAC using {forecast_type} forecasts and sell_price_ratio={sell_price_ratio}")
        model = SAC(**sac_params)
        model.learn(total_timesteps=total_timesteps, callback=ProgressBarCallback())
        if model_save_path is not None:
            model.save(model_save_path)

    print(f"Testing SAC using {forecast_type} forecasts and sell price ratio: {sell_price_ratio}")
    # Create stacked testing environment
    def make_test_env():
        return BatteryEnv(
            test["prices"], test[forecast_col], test["generation"], test["demand"], test["weather"],
            battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
            discrete_actions=False, sell_price_ratio=sell_price_ratio
        )
    vec_test_env = make_vec_env(make_test_env, n_envs=1, seed=SEED)
    if n_stack > 1:
        stacked_test_env = VecFrameStack(vec_test_env, n_stack=n_stack)
    else:
        stacked_test_env = vec_test_env

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

    if evaluation:
        # Evaluate RL results
        no_batt_curve_rl, batt_profit_curve_rl, savings_rl, savings_pct = evaluate_strategy(
            df_full=simdata,
            df_battery_result=df_rl_result_2024,
            start="2024-01-01",
            end="2024-12-31 23:00"
        )
    else:
        no_batt_curve_rl = batt_profit_curve_rl = savings_rl = savings_pct = None


    return df_rl_result_2024, no_batt_curve_rl, batt_profit_curve_rl, savings_rl, savings_pct