import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def preprocess_energy_data(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.name = 'Timestamp'
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')

    # Handle duplicates
    if df.index.duplicated().sum() > 0:
        df = df.sort_index().groupby(level=0).mean()

    # Force hourly frequency and interpolate missing values
    df = df.asfreq('h').interpolate(limit_direction='both')

    # Target variable
    price_series = df['Price'].dropna()

    # Complete, naive all exogenous variables
    # base_vars = [
    #     'Solar', 'Wind Onshore', 'Wind Offshore',
    #     'Load Forecast (MW)', 'Actual Aggregated',
    #     'Biomass', 'Fossil Gas', 'Nuclear', 'Other', 'Waste'
    # ]
    # weather_vars = [
    #     'temperature_2m', 'cloudcover', 'wind_speed_10m',
    #     'shortwave_radiation', 'sunshine_duration', 'windgusts_10m'
    # ]

    base_vars = [
        'Solar', 'Wind Onshore', 'Wind Offshore',
        'Load Forecast (MW)', 'Actual Aggregated', 'Fossil Gas', 'Other', 'Waste'
    ]
    weather_vars = [
        'temperature_2m', 'cloudcover', 'wind_speed_10m',
        'shortwave_radiation', 'sunshine_duration'
    ]
    
    exog_vars = base_vars + weather_vars
    exog = df[exog_vars].copy()

    # Add time-based features
    df['Hour'] = df.index.hour
    df['IsWeekend'] = (df.index.dayofweek >= 5).astype(float)
    df['Month'] = df.index.month
    month_dummies = pd.get_dummies(df['Month'], prefix='Month', drop_first=True)
    extra_features = pd.concat([df[['Hour', 'IsWeekend']], month_dummies], axis=1).astype(float)

    # Combine and interpolate
    exog_full = pd.concat([exog, extra_features], axis=1).interpolate(limit_direction='both')
    exog_raw = exog_full

    # Scale features
    scaler = StandardScaler()
    exog_scaled = pd.DataFrame(
        scaler.fit_transform(exog_full),
        columns=exog_full.columns,
        index=exog_full.index
    )

   # Scale price
    scaler_price = StandardScaler()
    price_scaled = pd.Series(
        scaler_price.fit_transform(price_series.values.reshape(-1, 1)).flatten(),
        index=price_series.index,
        name="Price"
    )

    # Align indices
    common_index = price_series.index.intersection(exog_scaled.index)
    price_series = price_series.loc[common_index]
    price_scaled = price_scaled.loc[common_index]
    exog_scaled = exog_scaled.loc[common_index]
    price_series_kWh = price_series / 1000  # Convert to €/kWh

    return price_series, price_scaled, exog_scaled, exog_raw, price_series_kWh

def get_descriptive(price, exog_raw):
    """
    Compute and display descriptive statistics for selected columns.
    """
    # Specify selected columns
    selected_cols = [
        'Price',
        'Solar',
        'Wind Onshore',
        'Wind Offshore',
        'Load Forecast (MW)',
        'Actual Aggregated',
        'temperature_2m',
        'cloudcover',
        'sunshine_duration'
    ]
    # Combine the price series with the exogenous variables
    full_data = pd.concat([price.rename('Price'), exog_raw], axis=1)
    full_data = full_data[selected_cols]

    # Compute descriptive statistics
    desc_stats = full_data.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    # Round for readability
    desc_stats_rounded = desc_stats.round(2)

    # Display as a clean table
    display(desc_stats_rounded)

def pricevisualization(price):
    """
    Visualize electricity price diagnostics: time series, volatility, distributions, ACF, PACF.
    """
    import matplotlib.pyplot as plt

    # Filter last year
    last_year = price.last('365D')
    returns = np.log(last_year / last_year.shift(1)).dropna()

    # Run ADF test for stationarity
    adf_result = adfuller(price.dropna())
    print("ADF Test Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical Values:", adf_result[4])
    if adf_result[1] < 0.05:
        print("The series is likely stationary (reject H0).")
    else:
        print("The series is likely non-stationary (fail to reject H0).")

    # First: Time Series plot
    plt.figure(figsize=(14, 4))
    plt.plot(price.index, price, label='Price (€/MWh)', linewidth=0.7)
    plt.title('Hourly Electricity Prices of all data (2022-2024)', fontsize=14)
    plt.ylabel('€/MWh')
    plt.xlabel('Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Second: Price Distribution (Histogram + KDE) and Log Return Distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.histplot(last_year, bins=80, kde=True, ax=axes[0])
    axes[0].set_title('Price Distribution (Histogram + KDE)')
    sns.histplot(returns, bins=80, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Log Return Distribution')
    plt.tight_layout()
    plt.show()

    # Third: ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(last_year.dropna(), lags=48, ax=axes[0])
    axes[0].set_title('Autocorrelation (ACF) – 2 days')
    plot_pacf(last_year.dropna(), lags=48, method='ywm', ax=axes[1])
    axes[1].set_title('Partial Autocorrelation (PACF) – 2 days')
    plt.tight_layout()
    plt.show()


def correlation_analysis(price, exog, threshold=0.95):
    """
    Analyze and visualize correlations between price and exogenous variables.
    Shows a heatmap, sorted correlations with price, and highly collinear pairs.
    """

    # Combine price and exogenous variables for correlation analysis
    corr_df = pd.concat([price.rename('Price'), exog], axis=1).dropna()

    # Drop all columns that start with 'Month'
    corr_df = corr_df.loc[:, ~corr_df.columns.str.startswith('Month')]

    # Compute correlation matrix
    corr_matrix = corr_df.corr()

    # Plot heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        annot=False,
        fmt=".2f",
        cbar_kws={'label': 'Correlation'}
    )
    plt.title("Correlation Heatmap: Price and Exogenous Variables")
    plt.tight_layout()
    plt.show()

    # Show correlations with price, sorted
    print("Correlation of each variable with Price:")
    print(corr_matrix['Price'].sort_values(ascending=False))

    # Find highly collinear pairs
    corrs = corr_matrix.abs()
    high_corr_pairs = [
        (corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
        for i in range(len(corr_matrix))
        for j in range(i + 1, len(corr_matrix))
        if corrs.iloc[i, j] > threshold
    ]
    for var1, var2, corr_val in high_corr_pairs:
        print(f"High collinearity: {var1} & {var2} (corr={corr_val:.2f})")


def generate_simdata(price, seed=2025, pv_peak_kw=50.0, demand_base_kw=20.0, demand_peak_kw=60.0):
    """
    Generate simulated PV generation and demand data, and align with given price series.
    Returns a DataFrame with columns: generation_kWh, demand_kWh, price, buy_price, sell_price.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate a continuous hourly index from 2022-01-01 to 2024-12-31 23:00 (no timezone yet)
    raw_dates = pd.date_range(start="2022-01-01", end="2024-12-31 23:00", freq="h")

    def simulate_pv(dates, peak_kw: float = 50.0) -> np.ndarray:
        hours = dates.hour
        # Day-of-day sine-shape (0 before 6:00, peak at 12:00, 0 after 18:00)
        day_factor = np.clip(np.sin((hours - 6) * np.pi / 12), 0, 1)
        # Seasonal factor: peak around day 172 (≈June 21), floor at 0.2 in mid-winter
        doy = dates.dayofyear
        seasonal_factor = np.clip(np.cos((doy - 172) * 2 * np.pi / 365), 0.2, 1)
        # Add small noise
        noise = np.random.normal(0, 0.1, size=len(dates))
        pv = peak_kw * day_factor * seasonal_factor * (1 + noise)
        return np.clip(pv, 0.0, None)

    def simulate_demand(dates, base_kw: float = 20.0, peak_kw: float = 60.0) -> np.ndarray:
        demand_values = []
        for ts in dates:
            hour = ts.hour
            weekday = ts.weekday()  # 0=Monday, …, 6=Sunday
            if (8 <= hour <= 17) and (weekday < 5):
                # Work hours on weekdays
                val = np.random.normal(loc=peak_kw, scale=5.0)
            else:
                # Off-hours or weekend
                val = np.random.normal(loc=base_kw, scale=3.0)
            demand_values.append(val)
        return np.clip(demand_values, 0.0, None)

    simdata = pd.DataFrame(index=raw_dates)
    simdata["generation_kWh"] = simulate_pv(raw_dates, peak_kw=pv_peak_kw)
    simdata["demand_kWh"]    = simulate_demand(raw_dates, base_kw=demand_base_kw, peak_kw=demand_peak_kw)

    ambiguous_flags = pd.Series(False, index=raw_dates)
    simdata.index = raw_dates.tz_localize(
        "Europe/Amsterdam",
        ambiguous=ambiguous_flags,
        nonexistent='shift_forward'
    )

    simdata = simdata[~simdata.index.isna()].sort_index()
    simdata = simdata.interpolate(method="time", limit_direction="both")

    simdata = simdata.join(price.rename("price"), how="left")

    simdata["price"] = simdata["price"].ffill().bfill()

    missing_price = simdata["price"].isna().sum()
    if missing_price:
        print(f"⚠️  Warning: {missing_price} missing price values; interpolating...")
        simdata["price"] = simdata["price"].interpolate(method="time", limit_direction="both")

    simdata["buy_price"]  = simdata["price"] / 1000.0       # €/kWh
    simdata["sell_price"] = simdata["buy_price"] * 0.30     # 30% of buy-price

    simdata = simdata[~simdata.index.duplicated(keep="first")]

    return simdata


def visualize_simdata(simdata):
    """
    Visualize simulated demand and PV generation for a week in June and November.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Select two weeks
    june_start = "2022-06-13"
    june_end   = "2022-06-19 23:00"
    nov_start  = "2022-11-14"
    nov_end    = "2022-11-20 23:00"

    sim_june = simdata.loc[june_start:june_end]
    sim_nov  = simdata.loc[nov_start:nov_end]

    # Create x-axis: 0–167 hours, then convert to day labels
    x = range(168)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    tick_positions = [i * 24 for i in range(7)]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # June
    ax1.plot(x, sim_june["demand_kWh"].values, label="Demand (kWh)", color="dodgerblue")
    ax1.plot(x, sim_june["generation_kWh"].values, label="PV Generation (kWh)", color="orange")
    ax1.set_title("Simulated Demand and Generation – Week in June")
    ax1.set_ylabel("kWh")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # November
    ax2.plot(x, sim_nov["demand_kWh"].values, label="Demand (kWh)", color="dodgerblue")
    ax2.plot(x, sim_nov["generation_kWh"].values, label="PV Generation (kWh)", color="orange")
    ax2.set_title("Simulated Demand and Generation – Week in November")
    ax2.set_ylabel("kWh")
    ax2.set_xlabel("Day of the Week")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    # Set x-ticks to day names
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(day_labels)

    plt.tight_layout()
    plt.show()


