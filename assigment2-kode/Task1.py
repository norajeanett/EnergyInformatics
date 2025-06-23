"""
Assignment 2 – Task 1
Single-predictor wind-power forecasting (WS10  →  POWER)

Models   : Linear Regression, k-Nearest Neighbour, Support-Vector Regression, ANN
Products : · Four forecast-template *.csv files (one per model)
           · Console RMSE table
           · Four overlay plots (truth vs model)

Authors    : <Bakke, Iselin Mordal; Gregussen, Andre Maharaj;
                Tønnessen, Nora Jeanett; Vik, Henrik Halse.>
Created   : 2025-04-28
"""

# IMPORTS
import pathlib
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model    import LinearRegression
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.svm             import SVR
from sklearn.neural_network  import MLPRegressor
from sklearn.pipeline        import make_pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_squared_error

#  FILE LOCATIONS
#  ➜ Adjust these paths if your directory layout differs
PATH_TRAIN     = pathlib.Path("TrainData.csv")
PATH_WEATHER   = pathlib.Path("WeatherForecastInput.csv")
PATH_SOLUTION  = pathlib.Path("Solution.csv")
PATH_TEMPLATE  = pathlib.Path("ForecastTemplate.csv")

#  ➜ Output folder for model forecasts
OUT_LR   = pathlib.Path("Task1_results/ForecastTemplate1-LR.csv")
OUT_kNN  = pathlib.Path("Task1_results/ForecastTemplate1-kNN.csv")
OUT_SVR  = pathlib.Path("Task1_results/ForecastTemplate1-SVR.csv")
OUT_ANN  = pathlib.Path("Task1_results/ForecastTemplate1-NN.csv")

# 1. READ DATA
train      = pd.read_csv(PATH_TRAIN)       # 01-Jan-2012 … 31-Oct-2013
weather_in = pd.read_csv(PATH_WEATHER)     # WS10 forecasts for Nov-2013
truth      = pd.read_csv(PATH_SOLUTION)    # True power for Nov-2013
template   = pd.read_csv(PATH_TEMPLATE)    # Blank submission template

# ▸ Prepare design matrices
X_train = train[["WS10"]].values           # predictor  : wind-speed (m s⁻¹)
y_train = train["POWER"].values            # target     : normalised power
X_eval  = weather_in[["WS10"]].values      # data to forecast
y_eval  = truth["POWER"].values            # ground truth (for RMSE only)

# 2. DEFINE & FIT MODELS
# Each entry is {name: estimator}.  Pipelines handle scaling where needed.
models = {
    "LR":  LinearRegression(),                             # global linear fit
    "kNN": KNeighborsRegressor(n_neighbors=5),             # local averaging
    "SVR": make_pipeline(StandardScaler(),                 # ε-SVR + RBF kernel
                         SVR(kernel="rbf",
                             C=10.0,
                             epsilon=0.01)),
    "ANN": make_pipeline(StandardScaler(),                 # 2×20-ReLU MLP
                         MLPRegressor(hidden_layer_sizes=(20, 20),
                                      activation="relu",
                                      solver="adam",
                                      max_iter=2000,
                                      random_state=0))
}

pred = {}                                                  # hold model outputs
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    pred[name] = mdl.predict(X_eval)

# 3. SAVE FORECAST FILES
# Re-use the template, overwrite “FORECAST”, then save to model-specific file
template["FORECAST"] = pred["LR"]
template.to_csv(OUT_LR, index=False)

template["FORECAST"] = pred["kNN"]
template.to_csv(OUT_kNN, index=False)

template["FORECAST"] = pred["SVR"]
template.to_csv(OUT_SVR, index=False)

template["FORECAST"] = pred["ANN"]
template.to_csv(OUT_ANN, index=False)

# 4. EVALUATE & REPORT
rmse = {name: np.sqrt(mean_squared_error(y_eval, p))
        for name, p in pred.items()}

rmse_tbl = (pd.DataFrame.from_dict(rmse, orient="index", columns=["RMSE"])
              .sort_values("RMSE"))

print("\nRMSE – Wind-farm normalised power (Nov-2013)\n" + "-"*45)
print(rmse_tbl.to_string(float_format="%.6f"))

# 5. PLOT OVERLAYS
# One figure per model: black = truth, colour = prediction
ts = pd.to_datetime(truth["TIMESTAMP"])    # convert string → datetime

for name, p in pred.items():
    plt.figure(figsize=(10, 4))
    plt.plot(ts, y_eval, "k-",  label="True")
    plt.plot(ts, p,      label=f"Predicted · {name}")
    plt.xlabel("Timestamp")
    plt.ylabel("Normalised power")
    plt.title(f"One-hour-ahead forecast – {name}")
    plt.legend()
    plt.tight_layout()
    plt.show()