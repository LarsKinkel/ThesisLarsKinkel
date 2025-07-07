import gymnasium as gym
import numpy as np
import pandas as pd

class BatteryEnv(gym.Env):
    def __init__(self, prices, prices_hat, generation, demand, weather,
                 battery_capacity=200.0, charge_limit=100.0, discharge_limit=100.0, efficiency=0.98, initial_soc=0,
                 discrete_actions=True, sell_price_ratio=1.0):
        super(BatteryEnv, self).__init__()
        self.weather = weather
        self.prices = prices / 1000.0  # Convert to €/kWh
        self.prices_hat = prices_hat / 1000.0  # Convert to €/kWh
        self.sell_price_ratio = sell_price_ratio
        self.generation = generation
        self.demand = demand
        self.capacity = battery_capacity
        self.charge_limit = charge_limit
        self.discharge_limit = discharge_limit
        self.efficiency = efficiency
        self.soc = initial_soc
        self.initial_soc = initial_soc  # Store initial SOC for reset
        self.t = 0

        self.discrete = discrete_actions
        
        # Define action and observation spaces
        if self.discrete:
            self.action_space = gym.spaces.Discrete(7)
        else:
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(42,), dtype=np.float32)  # Adjusted for state size
        
        # Tracking metrics
        self.actions = []
        self.soc_list = []

    def reset(self, seed=None, options=None):
        self.t = 0
        self.actions = []
        self.soc = self.initial_soc 
        self.soc_list = []
        return self._get_state(), {}

    def step(self, action):
        price = self.prices.iloc[self.t]
        buy_price = price  # €/kWh
        sell_price = price * self.sell_price_ratio # €/kWh

        reward = 0
        profit = 0
        initial_soc = self.soc  # Store SOC before the step
        grid_energy_bought = 0  # Energy bought from the grid
        grid_energy_sold = 0  # Energy sold to the grid
        charge_from_battery = 0  # Energy charged into the battery
        discharge_from_battery = 0  # Energy discharged from the battery

        demand = self.demand.iloc[self.t]
        generation = self.generation.iloc[self.t]

        # Fully charge the battery using generated energy
        charge_from_generation = min(generation, self.capacity - self.soc)
        self.soc += charge_from_generation * self.efficiency
        charge_from_battery += charge_from_generation  # Track energy charged into the battery
        generation -= charge_from_generation

        ########## Handle excess generation and demand ##########
        # Route excess generation to meet demand
        pv_to_demand = min(generation, demand)
        demand -= pv_to_demand
        generation -= pv_to_demand

        # Sell remaining excess generation to the grid
        if generation > 0:
            reward += generation * sell_price
            profit += generation * sell_price
            grid_energy_sold += generation  # Track energy sold to the grid
            generation = 0

        ############ Handle actions ##########
        if self.discrete:
            if action == 0:  # Max charge from grid
                charge_amount = min(self.charge_limit, self.capacity - self.soc)
                self.soc += charge_amount * self.efficiency
                reward -= charge_amount * buy_price  # Penalize charging from the grid
                profit -= charge_amount * buy_price  # Penalize charging from the grid
                grid_energy_bought += charge_amount  # Track energy bought from the grid

                unmet_demand = max(0, demand)
                reward -= unmet_demand * buy_price  # Penalize unmet demand
                profit -= unmet_demand * buy_price
                grid_energy_bought += unmet_demand  # Track energy bought to meet unmet demand
                demand = 0  

            elif action == 1:  # Max Half charge
                charge_amount = min(self.charge_limit / 2, self.capacity - self.soc)
                self.soc += charge_amount * self.efficiency
                reward -= charge_amount * buy_price  # Penalize charging from the grid
                profit -= charge_amount * buy_price
                grid_energy_bought += charge_amount  # Track energy bought from the grid

                unmet_demand = max(0, demand)
                reward -= unmet_demand * buy_price  # Penalize unmet demand
                profit -= unmet_demand * buy_price
                grid_energy_bought += unmet_demand  # Track energy bought to meet unmet demand
                demand = 0

            elif action == 2:  # Idle
                unmet_demand = max(0, demand)
                reward -= demand * buy_price
                profit -= demand * buy_price 
                grid_energy_bought += demand  # Track energy bought to meet unmet demand

            elif action == 3:  # Max Half discharge to demand
                discharge_amount = min(self.discharge_limit / 2, self.soc * self.efficiency)
                discharge_to_demand = min(discharge_amount, demand)
                self.soc -= discharge_to_demand / self.efficiency
                discharge_from_battery += discharge_to_demand  # Track energy discharged from the battery

                unmet_demand = max(0, demand - discharge_to_demand)
                reward -= unmet_demand * buy_price  # Penalize grid usage for meeting demand
                profit -= unmet_demand * buy_price
                grid_energy_bought += unmet_demand  # Track energy bought to meet unmet demand
                demand = 0

            elif action == 4:  # Max discharge to demand
                discharge_amount = min(self.discharge_limit, self.soc * self.efficiency)
                discharge_to_demand = min(discharge_amount, demand)
                self.soc -= discharge_to_demand / self.efficiency
                discharge_from_battery += discharge_to_demand  # Track energy discharged from the battery

                unmet_demand = max(0, demand - discharge_to_demand)
                reward -= unmet_demand * buy_price  # Penalize grid usage for meeting demand
                profit -= unmet_demand * buy_price
                grid_energy_bought += unmet_demand  # Track energy bought to meet unmet demand
                demand = 0

            elif action == 5:  # Half discharge to grid
                discharge_amount = min(self.discharge_limit / 2, self.soc * self.efficiency)
                self.soc -= discharge_amount / self.efficiency
                discharge_from_battery += discharge_amount  # Track energy discharged from the battery

                reward += discharge_amount * sell_price  # Reward for selling energy to the grid
                profit += discharge_amount * sell_price
                grid_energy_sold += discharge_amount  # Track energy sold to the grid
                reward -= demand * buy_price  # Penalize unmet demand
                profit -= demand * buy_price
                grid_energy_bought += demand  # Track energy bought to meet unmet demand
                demand = 0

            elif action == 6:  # Full discharge to grid
                discharge_amount = min(self.discharge_limit, self.soc * self.efficiency)
                self.soc -= discharge_amount / self.efficiency
                discharge_from_battery += discharge_amount  # Track energy discharged from the battery

                reward += discharge_amount * sell_price  # Reward for selling energy to the grid
                profit += discharge_amount * sell_price
                grid_energy_sold += discharge_amount  # Track energy sold to the grid
                reward -= demand * buy_price # Penalize unmet demand
                profit -= demand * buy_price
                grid_energy_bought += demand  # Track energy bought to meet unmet demand
                demand = 0
        else:
            # Pick the dominant intention
            if isinstance(action, np.ndarray):
                action = np.squeeze(action)
            if action.shape == ():  # still scalar
                action = np.array([action] * 3)
            a_grid_chg, a_grid_dis, a_dem_dis = action
            winner = int(np.argmax(action))  # 0: charge from grid, 1: discharge to grid, 2: discharge to demand

            # Translate to a power command
            if winner == 0:         # ---------- charge from grid ---------
                p_cmd = +a_grid_chg * self.charge_limit
                target_grid = True
            elif winner == 1:       # ---------- discharge to grid ---------
                p_cmd = -a_grid_dis * self.discharge_limit
                target_grid = True
            else:                   # ---------- discharge to demand -------
                p_cmd = -a_dem_dis * self.discharge_limit
                target_grid = False  # First tries to cover demand

            # Safety clamp
            p_cmd = np.clip(p_cmd, -self.discharge_limit, self.charge_limit)
            # ------------------------------------------------------------------
            # 3) Run the accounting logic using `p_cmd`
            # ------------------------------------------------------------------
            if p_cmd >= 0:  # ======== CHARGE ===================
                real_charge = min(p_cmd, self.capacity - self.soc)
                self.soc += real_charge * self.efficiency
                grid_energy_bought += real_charge
                reward -= real_charge * buy_price
                profit -= real_charge * buy_price

                # Any unmet demand has to be bought from the grid
                unmet_demand = max(0, demand)
                reward -= unmet_demand * buy_price
                profit -= unmet_demand * buy_price
                grid_energy_bought += unmet_demand
                demand = 0

            else:  # ======== DISCHARGE ====================
                discharge = min(-p_cmd, self.soc * self.efficiency)

                if target_grid:  # ---------- sell to grid ----------
                    self.soc -= discharge / self.efficiency
                    reward += discharge * sell_price
                    profit += discharge * sell_price
                    grid_energy_sold += discharge

                    # Grid still needs to cover remaining demand
                    reward -= demand * buy_price
                    profit -= demand * buy_price
                    grid_energy_bought += demand
                    demand = 0
                else:  # ---------- meet demand ----------
                    to_dem = min(discharge, demand)
                    self.soc -= to_dem / self.efficiency
                    discharge_from_battery += to_dem

                    unmet_demand = demand - to_dem
                    reward -= unmet_demand * buy_price
                    profit -= unmet_demand * buy_price
                    grid_energy_bought += unmet_demand
                    demand = 0

        ##### Reward shaping
        # peak_timing_bonus = 1  # Bonus for selling at peak times

        # # Penalize selling too early (according to forecasts)
        # if action in [5, 6]:  # Discharge to grid
        #     future_peak = np.max(self.prices_hat[:6])
        #     if future_peak > price:
        #         penalty = future_peak - price
        #         reward -= discharge_amount * penalty
        # # Penalize charging too early (according to forecasts)
        # if action in [0, 1]:  # Charging from grid
        #     future_min = np.min(self.prices_hat[:6])
        #     if future_min < price:
        #         penalty = price - future_min
        #         reward -= charge_amount * penalty
        # # Encourage peak timing for selling
        # if action in [5, 6]:
        #     peak_t = np.argmax(self.prices_hat[:6])
        #     if peak_t <= 1:
        #         reward += peak_timing_bonus * (6 - peak_t) / 6


        # Ensure SOC is within realistic bounds
        self.soc = max(0, min(self.soc, self.capacity))

        # Calculate the difference in SOC
        delta_soc = self.soc - initial_soc

        self.t += 1
        terminated = self.t >= len(self.prices) - 1
        truncated = False
        self.actions.append(action)
        self.soc_list.append(self.soc)

        # Return state, reward, termination status, and metrics
        return self._get_state(), reward, terminated, truncated, {
            "total_demand": self.demand.iloc[self.t - 1],
            "total_generation": self.generation.iloc[self.t - 1],
            "delta_soc": delta_soc,
            "grid_energy_bought": grid_energy_bought,
            "grid_energy_sold": grid_energy_sold,
            "charge_from_battery": charge_from_battery,
            "discharge_from_battery": discharge_from_battery,
            "profit": profit,
        }

    def _get_state(self):
        price = self.prices.iloc[self.t]
        soc_norm = self.soc / self.capacity
        price_norm = price / self.prices.max()  # Normalize price to its maximum value
        hour = self.t % 24
        sin_hour = np.sin(2 * np.pi * hour / 24)
        cos_hour = np.cos(2 * np.pi * hour / 24)

        demand_norm = self.demand.iloc[self.t] / self.demand.max()  # Normalize demand to its maximum value
        generation_norm = self.generation.iloc[self.t] / self.generation.max()  # Normalize generation to its maximum value
        # Price forecast
        prices_hat = self.prices_hat[self.t:self.t+24] / self.prices_hat.max()  # Normalize price forecast to its maximum value
        if len(prices_hat) < 24:
            prices_hat = np.pad(prices_hat.to_numpy(), (0, 24 - len(prices_hat)), mode='constant')
        else:
            prices_hat = prices_hat.to_numpy().flatten()

        # Price-based features (forecast-aware)
        price_trend = prices_hat[6] - prices_hat[0]
        price_mean_6h = np.mean(prices_hat[:6])
        price_peak_6h = np.max(prices_hat[:6])
        price_peak_later = int(np.argmax(prices_hat) > 6)
        price_above_now = price_peak_6h > price_norm

        # Forecast trend features
        trend_3h = prices_hat[3] - price_norm if len(prices_hat) > 3 else 0
        trend_6h = prices_hat[6] - price_norm if len(prices_hat) > 6 else 0
        trend_mean = np.mean(prices_hat[:6]) - price_norm
        peak_hour = np.argmax(prices_hat) / 24


        # Optional: Add weather forecasts (replace `weather_df` with actual DataFrame passed into env)
        shortwave_norm = self.weather['shortwave_radiation'].iloc[self.t:self.t+6].mean() / self.weather['shortwave_radiation'].max()
        cloudcover_norm = self.weather['cloudcover'].iloc[self.t:self.t+6].mean() / self.weather['cloudcover'].max()
        temperature_norm = self.weather['temperature_2m'].iloc[self.t:self.t+6].mean() / self.weather['temperature_2m'].max()

        state = np.concatenate([
            np.array([
                soc_norm, price_norm, sin_hour, cos_hour, demand_norm, generation_norm, 
                shortwave_norm, cloudcover_norm, temperature_norm,
                trend_3h, trend_6h, trend_mean, peak_hour,
                price_trend, price_mean_6h, price_peak_6h, price_peak_later, price_above_now
            ]),
            prices_hat  # 24 values
        ])

        return state