import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AutoReg
from helpers import *


# Energy price model functions

# Baseline AR model
def rolling_forecast_AR(series, horizon=24, days=60, lags=24):
    last = series.index[-1].normalize()
    first = last - pd.Timedelta(days=days-1)
    RMSE, MAE, T = [], [], []
    dates = list(pd.date_range(first, last, freq='D'))
    all_preds = []

    for d in tqdm(dates, desc="AR Forecast"):
        y_train = series.loc[:d-pd.Timedelta(hours=1)]
        y_test  = series.loc[d:d+pd.Timedelta(hours=horizon-1)]
        if len(y_train)<lags or len(y_test)<horizon: continue
        mdl = AutoReg(y_train,lags=lags,old_names=False).fit()
        y_hat = mdl.predict(start=len(y_train), end=len(y_train)+horizon-1)
        y_hat.index = y_test.index
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_hat)))
        MAE.append(mean_absolute_error(y_test, y_hat))
        T.append(d)
        all_preds.append(y_hat)

    preds_full = pd.concat(all_preds).sort_index() if all_preds else pd.Series(dtype=float)

    return RMSE, MAE, T, preds_full

# ARX model extension, including exogenous variables
def rolling_forecast_ARX(series, exog, horizon=24, days=60, lags=24):
    last = series.index[-1].normalize()
    first = last - pd.Timedelta(days=days-1)
    RMSE, MAE, T, all_preds = [], [], [], []
    dates = list(pd.date_range(first, last, freq='D'))

    for d in tqdm(dates, desc="ARX Forecast"):
        y_train = series.loc[:d-pd.Timedelta(hours=1)]
        X_train = exog.loc[:d-pd.Timedelta(hours=1)]
        y_test  = series.loc[d:d+pd.Timedelta(hours=horizon-1)]
        X_test  = exog.loc[d:d+pd.Timedelta(hours=horizon-1)]
        
        if len(y_train)<lags or len(y_test)<horizon: continue
        scl = StandardScaler(); X_tr = scl.fit_transform(X_train); X_te = scl.transform(X_test)
        mdl = AutoReg(y_train,lags=lags,exog=X_tr,old_names=False).fit()
        y_hat = mdl.predict(start=len(y_train), end=len(y_train)+horizon-1, exog_oos=X_te)
        y_hat.index = y_test.index
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_hat)))
        MAE.append(mean_absolute_error(y_test, y_hat))
        T.append(d)
        all_preds.append(y_hat)
    
    preds_full = pd.concat(all_preds).sort_index() if all_preds else pd.Series(dtype=float)

    return RMSE, MAE, T, preds_full

def rolling_forecast_LEAR_with_AR_fallback(series, exog, horizon=24, days=60,
                                           window_days=56, cv_folds=5,
                                           max_pred_threshold=800,
                                           fallback_lags=24):
    from helpers import _asinh_transform, _asinh_inverse

    y_as, med, mad = _asinh_transform(series)

    feats = pd.DataFrame(index=series.index)
    feats['p_d1'] = y_as.shift(24)
    feats['p_d2'] = y_as.shift(48)
    feats['p_d3'] = y_as.shift(72)
    feats['p_d7'] = y_as.shift(168)

    for col in exog.columns:
        feats[f'{col}_d0'] = exog[col]
        feats[f'{col}_d1'] = exog[col].shift(24)
        feats[f'{col}_d7'] = exog[col].shift(168)

    hour = pd.get_dummies(series.index.hour, prefix='hour')
    dow = pd.get_dummies(series.index.dayofweek, prefix='dow')
    feats = pd.concat([feats, hour, dow], axis=1).ffill().bfill()

    last_day = series.index[-1].normalize()
    first_day = last_day - pd.Timedelta(days=days - 1)

    RMSE, MAE, DATES = [], [], []
    all_preds, all_true = [], []
    LEAR_used, AR_used = 0, 0
    model_used_per_day = {}

    for current_day in tqdm(pd.date_range(first_day, last_day, freq='D'), desc="LEAR + AR Forecast"):
        test_idx = pd.date_range(current_day, periods=horizon, freq='H')
        if test_idx[-1] not in series.index:
            print(f"Skipping {current_day.date()} — forecast horizon exceeds data")
            continue

        y_true = series.loc[test_idx]
        fallback = False

        tr_end = current_day - pd.Timedelta(hours=1)
        tr_st = tr_end - pd.Timedelta(days=window_days)
        train_idx = pd.date_range(tr_st, tr_end, freq='H')
        X_train = feats.loc[train_idx].dropna()

        if X_train.empty:
            fallback = True
        else:
            y_train = y_as.loc[X_train.index]
            try:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X_train)
                lasso = LassoCV(cv=TimeSeriesSplit(cv_folds), fit_intercept=True).fit(Xs, y_train)

                X_test = feats.loc[test_idx]
                X_hat = scaler.transform(X_test)
                y_hat_trans = lasso.predict(X_hat)
                preds = _asinh_inverse(pd.Series(y_hat_trans, index=test_idx), med, mad)

                if np.any(np.isnan(preds)) or np.any(np.abs(y_hat_trans) > max_pred_threshold):
                    raise ValueError("Invalid LEAR predictions")

                rmse = np.sqrt(mean_squared_error(y_true, preds))
                mae = mean_absolute_error(y_true, preds)

                if np.isnan(rmse) or rmse > 800:
                    raise ValueError("Unstable RMSE")

                LEAR_used += 1
                model_used_per_day[current_day.date()] = 'LEAR'
                y_hat = preds

            except Exception as e:
                print(f"LEAR fallback on {current_day.date()}: {e}")
                fallback = True

        if fallback:
            y_train = series.loc[:current_day - pd.Timedelta(hours=1)]
            if len(y_train) < fallback_lags or len(y_true) < horizon:
                print(f"Skipping {current_day.date()} — insufficient AR data")
                continue
            try:
                ar_model = AutoReg(y_train, lags=fallback_lags, old_names=False).fit()
                y_hat = ar_model.predict(start=len(y_train), end=len(y_train) + horizon - 1)
                y_hat.index = y_true.index
                rmse = np.sqrt(mean_squared_error(y_true, y_hat))
                mae = mean_absolute_error(y_true, y_hat)
                AR_used += 1
                model_used_per_day[current_day.date()] = 'AR'
            except Exception as e:
                print(f"AR error on {current_day.date()}: {e}")
                continue

        RMSE.append(rmse)
        MAE.append(mae)
        DATES.append(current_day)
        all_preds.append(y_hat)
        all_true.append(y_true)

    print(f"LEAR used on {LEAR_used} days, AR fallback used on {AR_used} days.")

    preds_full = pd.concat(all_preds).sort_index() if all_preds else pd.Series(dtype=float)
    true_full = pd.concat(all_true).sort_index() if all_true else pd.Series(dtype=float)

    return RMSE, MAE, DATES, preds_full, true_full, model_used_per_day


def EPFperformance(AR_t, AR_rmse, AR_mae, ARX_t, ARX_rmse, ARX_mae, LEAR_t, LEAR_rmse, LEAR_mae, LEAR_pred, true_prices):
    import matplotlib.pyplot as plt

    print("\n===== Average RMSE and MAE =====")
    print(f"              RMSE               MAE")
    print(f"AR         : {np.mean(AR_rmse):6.2f} €/MWh       {np.mean(AR_mae):6.2f} €/MWh")
    print(f"ARX        : {np.mean(ARX_rmse):6.2f} €/MWh       {np.mean(ARX_mae):6.2f} €/MWh")
    print(f"LEAR       : {np.mean(LEAR_rmse):6.2f} €/MWh       {np.mean(LEAR_mae):6.2f} €/MWh")

    plt.figure(figsize=(14,6))
    plt.plot(AR_t ,AR_rmse ,'-',label='AR')
    plt.plot(ARX_t,ARX_rmse,'-',label='ARX')
    plt.plot(LEAR_t,LEAR_rmse,'-',label='LEAR')
    plt.gca().set_ylim([0, 300])
    plt.ylabel('RMSE (€/MWh)')
    plt.xlabel('Date')
    plt.title('Daily RMSE – AR, ARX, LEAR')
    plt.grid(alpha=.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15,6))
    plt.plot(true_prices.index, true_prices, label="True Prices", alpha=0.7)
    plt.plot(LEAR_pred.index, LEAR_pred, label="Predicted Prices (LEAR)", alpha=0.7)
    plt.gca().set_ylim([-550, 800])
    plt.title("True vs Predicted Prices — LEAR")
    plt.xlabel("Time")
    plt.ylabel("Price (€/MWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def import_EPFforecasts(ar_path, lear_path, price, arx_path = None, tz="Europe/Amsterdam"):
    # Import AR forecasts
    AR_forecasts_pd = pd.read_csv(ar_path, index_col=0, parse_dates=True)
    AR_forecasts = AR_forecasts_pd.squeeze("columns")
    AR_forecasts.index = pd.to_datetime(AR_forecasts.index, utc=True).tz_convert(tz)

    ARX_forecasts = pd.Series(dtype=float)  # Initialize ARX as empty if not provided

    if arx_path is not None:
        # import ARX forecasts
        ARX_forecasts_pd = pd.read_csv(arx_path, index_col=0, parse_dates=True)
        ARX_forecasts = ARX_forecasts_pd.squeeze("columns")
        ARX_forecasts.index = pd.to_datetime(ARX_forecasts.index, utc=True).tz_convert(tz)  

    # Import LEAR forecasts
    LEAR_forecasts_pd = pd.read_csv(lear_path, index_col=0, parse_dates=True)
    LEAR_forecasts = LEAR_forecasts_pd.squeeze("columns")
    LEAR_forecasts.index = pd.to_datetime(LEAR_forecasts.index, utc=True).tz_convert(tz)

    # Perfect foresight forecasts
    perfect_foresight_forecasts = price.copy()

    # Find the latest start date among all series
    start_date = max(
        AR_forecasts.index.min(),
        LEAR_forecasts.index.min(),
        perfect_foresight_forecasts.index.min()
    )

    # Align the end date to the earliest end date
    end_date = min(
        AR_forecasts.index.max(),
        LEAR_forecasts.index.max(),
        perfect_foresight_forecasts.index.max()
    )

    # Align all series to this common range
    AR_forecasts = AR_forecasts.loc[start_date:end_date]
    ARX_forecasts = ARX_forecasts.loc[start_date:end_date]
    LEAR_forecasts = LEAR_forecasts.loc[start_date:end_date]
    perfect_foresight_forecasts = perfect_foresight_forecasts.loc[start_date:end_date]

    return AR_forecasts, ARX_forecasts, LEAR_forecasts, perfect_foresight_forecasts