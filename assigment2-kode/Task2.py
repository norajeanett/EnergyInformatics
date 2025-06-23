"""
Assignment 2 – Task 2
Dual-predictor wind-power forecasting
(WS10 & Wind-Direction  →  POWER)

Objective  : compare a single-input Linear-Regression baseline with a
             Multiple-Linear-Regression (WS10 + wind-direction) model.

Products   : • Task2_results/ForecastTemplate2.csv  (MLR forecast)
             • Task2_results/ForecastTemplate1-LR.csv (optional baseline)
             • Console RMSE table
             • Task2_results/Task2_comparison.png  (time-series overlay)

Authors    : Bakke, Iselin Mordal · Gregussen, Andre Maharaj ·
             Tønnessen, Nora Jeanett · Vik, Henrik Halse
Created    : 2025-04-29
"""

import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# helpers

def wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return meteorological wind-direction in degrees (0° = N, clockwise)."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def rmse(y_true, y_pred):
    """
    Version-agnostic RMSE:  √(mean-squared-error)
    Works with all scikit-learn releases.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def safe_save(template_path: str, out_path: str, forecast: np.ndarray) -> None:
    """Clone template, insert forecast, save – creating parent dir if needed."""
    dirpath = os.path.dirname(out_path)
    if dirpath:                                  # '' → current dir, skip mkdir
        os.makedirs(dirpath, exist_ok=True)

    df = pd.read_csv(template_path)
    df["FORECAST"] = forecast
    df.to_csv(out_path, index=False)
    print(f"→ {out_path} written ({len(df)} rows)")


#  data loaders

def load_training(csv_path: str) -> pd.DataFrame:
    """
    Read TrainData.csv and add a 'WindDir' column.
    No rows/columns are discarded – identical to the old helper.
    """
    df = pd.read_csv(csv_path)                      # 2012-01-01 … 2013-10-31
    df["WindDir"] = wind_direction(df["U10"], df["V10"])
    return df


def load_inputs(csv_path: str) -> pd.DataFrame:
    """
    Read WeatherForecastInput.csv and add the same 'WindDir' column.
    """
    df = pd.read_csv(csv_path)                      # 2013-11-01 … 2013-11-30
    df["WindDir"] = wind_direction(df["U10"], df["V10"])
    return df


def load_solution(csv_path: str) -> pd.DataFrame:
    sol = pd.read_csv(csv_path)
    sol["TIMESTAMP"] = pd.to_datetime(sol["TIMESTAMP"],
                                      format="%Y%m%d %H:%M")
    return sol


#  modelling

def fit_models(df: pd.DataFrame):
    y = df["POWER"].to_numpy()
    m_single = LinearRegression().fit(df[["WS10"]].to_numpy(),            y)
    m_multi  = LinearRegression().fit(df[["WS10", "WindDir"]].to_numpy(), y)
    return m_single, m_multi


#  evaluation & plotting

def print_rmse_table(truth: np.ndarray,
                     pred_lr: np.ndarray,
                     pred_mlr: np.ndarray) -> None:
    print("\nRMSE comparison – November 2013")
    print(" Model           RMSE")
    print("-----------------------")
    print(f" LR  (WS10)     {rmse(truth, pred_lr ):6.3f}")
    print(f" MLR (WS10+Dir) {rmse(truth, pred_mlr):6.3f}\n")


def plot_overlay(t: pd.Series,
                 truth: np.ndarray,
                 lr:    np.ndarray,
                 mlr:   np.ndarray,
                 fname: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(t, truth, label="True")
    plt.plot(t, lr,    label="LR (WS10)")
    plt.plot(t, mlr,   label="MLR (WS10+Dir)")
    plt.xlabel("Time"); plt.ylabel("Power (norm.)")
    plt.title("Wind-Power Forecast – November 2013")
    plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=300); plt.close()
    print(f"→ {fname} saved")


#  main orchestration

def run_task2() -> None:
    out_dir = "Task2_results"  # central results folder
    resource_dir = pathlib.Path(out_dir).parent
    os.makedirs(out_dir, exist_ok=True)

    # 1 training
    df_train = load_training("TrainData.csv")
    mdl_lr, mdl_mlr = fit_models(df_train)

    # 2 forecasting
    df_fc = load_inputs("WeatherForecastInput.csv")

    df_fc["TIMESTAMP"] = pd.to_datetime(  # <— NEW
        df_fc["TIMESTAMP"],
        format="%Y%m%d %H:%M")

    pred_lr = mdl_lr.predict(df_fc[["WS10"]].to_numpy())
    pred_mlr = mdl_mlr.predict(df_fc[["WS10", "WindDir"]].to_numpy())

    # 3 save forecasts
    safe_save("ForecastTemplate.csv",
              os.path.join(out_dir, "ForecastTemplate1-LR.csv"),
              pred_lr)

    safe_save("ForecastTemplate.csv",
              os.path.join(out_dir, "ForecastTemplate2.csv"),
              pred_mlr)

    # 4 evaluate
    sol_df = load_solution("Solution.csv")
    merged = sol_df.merge(df_fc[["TIMESTAMP"]]
                          .assign(LR=pred_lr, MLR=pred_mlr),
                          on="TIMESTAMP", how="inner")

    print_rmse_table(merged["POWER"].to_numpy(),
                     merged["LR"   ].to_numpy(),
                     merged["MLR"  ].to_numpy())

    # 5 visualisation
    plot_overlay(merged["TIMESTAMP"],
                 merged["POWER"].to_numpy(),
                 merged["LR"   ].to_numpy(),
                 merged["MLR"  ].to_numpy(),
                 os.path.join(out_dir, "Task2_comparison.png"))


if __name__ == "__main__":
    run_task2()
