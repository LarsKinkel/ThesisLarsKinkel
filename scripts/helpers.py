import numpy as np
import pandas as pd
import optuna
import random
import torch

def _asinh_transform(series):
    med = series.median()
    mad = (series - med).abs().median()*1.4826
    return np.arcsinh((series-med)/mad), med, mad

def _asinh_inverse(s, med, mad):
    return np.sinh(s)*mad + med

def compute_no_battery_cost(df):
    """
    Compute total electricity cost without any battery.
    """
    net_demand = df['demand_kWh'] - df['generation_kWh']
    
    cost_series_positive = net_demand.clip(lower=0) * (df['price'] / 1000)

    cost_series_negative = net_demand.clip(upper=0) * ((df['price']) / 1000)

    cost_series = cost_series_positive - cost_series_negative

    # Add cumulative cost to the DataFrame
    df = df.copy()
    df['no_battery_cost'] = cost_series.cumsum()

    return df['no_battery_cost'], cost_series.sum()

def compute_battery_cost(df_with_battery):
    """
    Compute total electricity cost *with* the battery
    """
    final_profit = df_with_battery['cumulative_profit'].iloc[-1]
    return -final_profit  # Because 'profit' tracks revenue - cost

def evaluate_strategy(df_full, df_battery_result, start="2024-01-01", end="2024-12-31"):
    df_period = df_full.loc[start:end].copy()
    df_battery = df_battery_result.loc[start:end]

    no_batt_curve, no_batt_cost = compute_no_battery_cost(df_period)
    batt_cost = compute_battery_cost(df_battery)

    savings = no_batt_cost - batt_cost
    savings_pct = (savings / no_batt_cost) * 100

    print(f"💰 No Battery Cost:         €{no_batt_cost:,.2f}")
    print(f"🔋 With Battery Cost:       €{batt_cost:,.2f}")
    print(f"✅ Savings from Battery:    €{savings:,.2f}")
    print(f"📉 Savings Percentage:       {savings_pct:.2f}%")

    return no_batt_curve, df_battery['cumulative_profit'], savings, savings_pct

def evaluate_battery_result(result_df, no_batt_curve, batt_profit_curve):

    fig, axs = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)

    # --- Plot 1: Cumulative cost comparison ---
    axs.plot(no_batt_curve.index, no_batt_curve.values, label='No Battery (Cost)', color='black')
    axs.plot(batt_profit_curve.index, -batt_profit_curve.values, label='With Battery (Cost)', color='green')
    axs.fill_between(no_batt_curve.index, no_batt_curve.values, -batt_profit_curve.values, 
                        where=(no_batt_curve > -batt_profit_curve), color='lightgreen', alpha=0.5, label='Savings')
    axs.set_title('Battery vs No Battery Costs', fontsize=16)
    axs.set_ylabel('€')
    axs.set_xlabel('Time')
    axs.legend()
    axs.grid(True)
    
    plt.show()



def align_and_split_data(price, simdata, AR_forecasts, LEAR_forecasts, perfect_foresight_forecasts, exog, 
                         train_end="2023-12-31 23:00:00", test_start="2024-01-01 00:00:00", tz="Europe/Amsterdam"):
    """
    Aligns all input dataframes/series to a common index and splits them into train and test sets.
    Returns dictionaries with aligned and split data.
    """

    aligned_start = AR_forecasts.index.min()  
    aligned_end = AR_forecasts.index.max()  

    aligned_prices = price.loc[aligned_start:aligned_end]
    aligned_generation = simdata["generation_kWh"].loc[aligned_start:aligned_end]
    aligned_demand = simdata["demand_kWh"].loc[aligned_start:aligned_end]
    aligned_AR = AR_forecasts.loc[aligned_start:aligned_end]
    aligned_LEAR = LEAR_forecasts.loc[aligned_start:aligned_end]
    aligned_perfect = perfect_foresight_forecasts.loc[aligned_start:aligned_end]

    common_index = aligned_prices.index
    aligned_generation = aligned_generation.reindex(common_index).interpolate()
    aligned_demand = aligned_demand.reindex(common_index).interpolate()
    aligned_weather = exog.reindex(common_index).interpolate()

    train_end = pd.Timestamp(train_end, tz=tz)
    test_start = pd.Timestamp(test_start, tz=tz)

    train = {
        "prices": aligned_prices.loc[aligned_start:train_end],
        "generation": aligned_generation.loc[aligned_start:train_end],
        "demand": aligned_demand.loc[aligned_start:train_end],
        "weather": aligned_weather.loc[aligned_start:train_end],
        "AR": aligned_AR.loc[aligned_start:train_end],
        "LEAR": aligned_LEAR.loc[aligned_start:train_end],
        "perfect": aligned_perfect.loc[aligned_start:train_end],
    }
    test = {
        "prices": aligned_prices.loc[test_start:aligned_end],
        "generation": aligned_generation.loc[test_start:aligned_end],
        "demand": aligned_demand.loc[test_start:aligned_end],
        "weather": aligned_weather.loc[test_start:aligned_end],
        "AR": aligned_AR.loc[test_start:aligned_end],
        "LEAR": aligned_LEAR.loc[test_start:aligned_end],
        "perfect": aligned_perfect.loc[test_start:aligned_end],
    }

    return train, test

def hypertuning(trial, traindata, testdata, simdata, algo="DQN", sellratio=1.0):
    from DQN import train_and_test_dqn
    from PPOd import train_and_test_PPOd
    from PPOc import train_and_test_PPOc
    from SAC import train_and_test_SAC

    SEED = 2025
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Sample hyperparameters for each algorithm
    if algo == "DQN":
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        gamma = trial.suggest_uniform("gamma", 0.90, 0.999)
        # timesteps = trial.suggest_int("timesteps", 50000, 200000)
        timesteps = 115000
        try:
            _, _, _, _, savings_pct = train_and_test_dqn(
                train=traindata,
                test=testdata,
                simdata=simdata,
                forecast_type="LEAR",
                sell_price_ratio=sellratio,
                load_existing_model=False,
                timesteps=timesteps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                evaluation = True
            )
            return savings_pct
        except Exception as e:
            print("DQN trial failed:", e)
            return -1e9

    elif algo == "PPOd":
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        gamma = trial.suggest_uniform("gamma", 0.90, 0.999)
        gae_lambda = trial.suggest_uniform("gae_lambda", 0.90, 1.0)
        n_steps = trial.suggest_int("n_steps", 32, 1024)
        # timesteps = trial.suggest_int("timesteps", 200_000, 2_000_000, step=200_000)
        timesteps = 1400000
        try:
            _, _, _, _, savings_pct = train_and_test_PPOd(
                train=traindata,
                test=testdata,
                simdata=simdata,
                forecast_type="LEAR",
                sell_price_ratio=sellratio,
                load_existing_model=False,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                n_steps=n_steps,
                total_timesteps=timesteps,
                evaluation = True
            )
            return savings_pct
        except Exception as e:
            print("PPOd trial failed:", e)
            return -1e9

    elif algo == "PPOc":
        # Use the same hyperparameters as PPOd, or adjust as needed
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        gamma = trial.suggest_uniform("gamma", 0.90, 0.999)
        gae_lambda = trial.suggest_uniform("gae_lambda", 0.90, 1.0)
        n_steps = trial.suggest_int("n_steps", 32, 1024)
        # timesteps = trial.suggest_int("timesteps", 200_000, 2_000_000, step=200_000)
        timesteps = 1800000
        try:
            _, _, _, _, savings_pct = train_and_test_PPOc(
                train=traindata,
                test=testdata,
                simdata=simdata,
                forecast_type="LEAR",
                sell_price_ratio=sellratio,
                load_existing_model=False,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                n_steps=n_steps,
                total_timesteps=timesteps,
                evaluation = True
            )
            return savings_pct
        except Exception as e:
            print("PPOc trial failed:", e)
            return -1e9
    elif algo == "SAC":
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        gamma = trial.suggest_uniform("gamma", 0.90, 0.999)
        tau = trial.suggest_loguniform("tau", 1e-5, 1e-2)
        timesteps = 500000  # Fixed for SAC, can be tuned if needed
        try:
            _, _, _, _, savings_pct = train_and_test_SAC(
                train=traindata,
                test=testdata,
                simdata=simdata,
                forecast_type="LEAR",
                sell_price_ratio=sellratio,
                load_existing_model=False,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                tau=tau,
                total_timesteps=timesteps,
                evaluation = True
            )
            return savings_pct
        except Exception as e:
            print("SAC trial failed:", e)
            return -1e9
        

    else:
        raise ValueError("Unknown algorithm: choose 'DQN', 'PPOd', or 'PPOc'.")
