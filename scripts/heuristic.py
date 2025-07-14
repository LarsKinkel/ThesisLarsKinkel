import numpy as np
import pandas as pd

def run_baseline_battery(df,
                         battery_capacity=200.0,
                         charge_limit=100.0,
                         discharge_limit=100.0,
                         efficiency=0.98,
                         initial_soc=0.0,
                         low_price_threshold=70.0,   # €/MWh
                         high_price_threshold=170.0,  # €/MWh
                         sellpriceratio = 1.0
                         ):

    soc = initial_soc
    profit = 0

    # Initialize tracking lists
    actions, soc_list, profit_list = [], [], []
    from_pv_list, from_grid_list, from_batt_list = [], [], []
    battery_charge_kWh_list, energy_sold_kWh_list = [], []

    for idx, row in df.iterrows():
        load = row["demand_kWh"]
        generation = row["generation_kWh"]
        market_price = row["price"]              # €/MWh
        buy_price = row["buy_price"]             # €/kWh
        sell_price = row["sell_price"]   / sellpriceratio        # €/kWh
        action = "idle"

        from_pv = from_grid = from_batt = 0.0
        battery_charge_kWh = energy_sold_kWh = 0.0

        # 1. Charge battery from PV (generation)
        pv_to_batt = min(generation, charge_limit, battery_capacity - soc)
        soc += pv_to_batt * efficiency
        battery_charge_kWh += pv_to_batt
        generation -= pv_to_batt

        # 2. Use remaining PV to serve demand
        from_pv = min(load, generation)
        load -= from_pv
        generation -= from_pv

        # 3. Sell any excess PV to the grid
        if generation > 0:
            profit += generation * sell_price
            energy_sold_kWh += generation
            action = "Sell PV to Grid"

        # 4. Meet remaining demand from battery
        if load > 0 and soc > 0:
            from_batt = min(load, discharge_limit, soc * efficiency)
            soc -= from_batt / efficiency
            load -= from_batt

        # 5. Meet remaining demand from grid
        if load > 0:
            from_grid = load
            profit -= from_grid * buy_price

        if market_price <= low_price_threshold:
            # Charge battery from grid
            charge_amount = min(charge_limit, battery_capacity - soc)
            soc += charge_amount * efficiency
            profit -= charge_amount * buy_price
            from_grid += charge_amount
            battery_charge_kWh = charge_amount
            action += " and Buy & Charge from Grid"

        elif market_price >= high_price_threshold and soc > 0:
            # Discharge battery to sell
            discharge_amount = min(discharge_limit, soc * efficiency)
            soc -= discharge_amount / efficiency
            profit += discharge_amount * sell_price
            energy_sold_kWh = discharge_amount
            action += "and Discharge & Sell to Grid"

        actions.append(action)
        soc_list.append(soc)
        profit_list.append(profit)
        from_pv_list.append(from_pv)
        from_grid_list.append(from_grid)
        from_batt_list.append(from_batt)
        battery_charge_kWh_list.append(battery_charge_kWh)
        energy_sold_kWh_list.append(energy_sold_kWh)

    df_result = df.copy()
    df_result["action"] = actions
    df_result["soc_kWh"] = soc_list
    df_result["cumulative_profit"] = profit_list
    df_result["from_pv"] = from_pv_list
    df_result["from_grid"] = from_grid_list
    df_result["from_batt"] = from_batt_list
    df_result["battery_charge_kWh"] = battery_charge_kWh_list
    df_result["energy_sold_kWh"] = energy_sold_kWh_list

    return df_result