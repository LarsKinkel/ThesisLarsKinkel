import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor
import os

from BatteryEnv import BatteryEnv # Import  custom environment

def train_and_test_dqn(
    train, test, simdata, forecast_type, sell_price_ratio=1.0, SEED=2025,
    model_save_path=None, load_existing_model=False, model_load_path=None,
    timesteps=50000, learning_rate=5e-4, batch_size=64, gamma=0.99, monitor=False, evaluation = False
):
    """
    Trains and tests a DQN agent on the provided train/test splits.
    If load_existing_model is True, loads the model from model_load_path and skips training.
    forecast_type: 'perfect', 'ar', or 'lear'
    Returns: DataFrame with test results, evaluation metrics.
    """
    from helpers import evaluate_strategy

    # Select forecast column based on forecast_type
    forecast_col = {
        "perfect": "perfect",
        "AR": "AR",
        "LEAR": "LEAR"
    }[forecast_type]

    if monitor == True:
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        monitor_filename = f"{log_dir}/DQN_{forecast_type}_monitor.csv"

        def make_monitored_train_env():
            env = BatteryEnv(
                train["prices"], train[forecast_col], train["generation"], train["demand"], train["weather"],
                battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
                discrete_actions=True, sell_price_ratio=sell_price_ratio)
            return Monitor(env, filename=monitor_filename)

        # Wrap the environment for compatibility with Stable-Baselines3
        vec_train_env = make_vec_env(lambda: make_monitored_train_env(), n_envs=1, seed=SEED)
    else:
        # Create training environment without monitoring
        train_env = BatteryEnv(
            train["prices"], train[forecast_col], train["generation"], train["demand"], train["weather"],
            battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
            discrete_actions=True, sell_price_ratio=sell_price_ratio)
        
        vec_train_env = make_vec_env(lambda: train_env, n_envs=1, seed=SEED)

    stacked_train_env = VecFrameStack(vec_train_env, n_stack=4)  # Stack 4 frames

    if load_existing_model and model_load_path is not None:
        print(f"Loading existing model from {model_load_path}")
        model = DQN.load(model_load_path)
    else:
        # Train the model
        print(f"The model is now Training DQN using {forecast_type} forecasts and sell_price_ratio={sell_price_ratio}")
        model = DQN(
            "MlpPolicy", stacked_train_env, verbose=0, learning_rate=learning_rate, buffer_size=50000, batch_size=batch_size,
            gamma=gamma, target_update_interval=500, seed=SEED
        )
        model.learn(total_timesteps=timesteps, callback=ProgressBarCallback())
        # Save the trained model
        if model_save_path is not None:
            model.save(model_save_path)

    print(f"The model is now Testing DQN using {forecast_type} forecasts and sell price ratio: {sell_price_ratio}")
    # Create testing environment
    test_env = BatteryEnv(
        test["prices"], test[forecast_col], test["generation"], test["demand"], test["weather"],
        battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
        discrete_actions=True, sell_price_ratio=sell_price_ratio)

    vec_test_env = make_vec_env(lambda: test_env, n_envs=1, seed=SEED)
    stacked_test_env = VecFrameStack(vec_test_env, n_stack=4)  # Stack 4 frames

    # Reset the environment
    obs = stacked_test_env.reset()
    done = False
    total_reward_rl = 0
    total_profit_rl = 0

    # Initialize tracking lists
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

    # Create a DataFrame for testing results
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