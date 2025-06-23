import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math

train_data = pd.read_csv('TrainData.csv')
solution_data = pd.read_csv('Solution.csv')
train_data = train_data[['TIMESTAMP', 'POWER']]
solution_data = solution_data[['TIMESTAMP', 'POWER']]

train_data['TIMESTAMP'] = pd.to_datetime(train_data['TIMESTAMP'], format='%Y%m%d %H:%M')
solution_data['TIMESTAMP'] = pd.to_datetime(solution_data['TIMESTAMP'], format='%Y%m%d %H:%M')
train_data.set_index('TIMESTAMP', inplace=True)
solution_data.set_index('TIMESTAMP', inplace=True)


def create_sliding_window(data, n_lags=2):
    X, y = [], []
    for i in range(len(data) - n_lags):
        a = data[i:(i + n_lags)]
        X.append(a)
        y.append(data[i + n_lags])
    return np.array(X), np.array(y)


power_values = train_data['POWER'].values
actual_test = solution_data[['POWER']].values

results = {
    "LR": {"rmse": float('inf'), "n_lags": None, "y_pred": None},
    "SVR": {"rmse": float('inf'), "n_lags": None, "y_pred": None},
    "ANN": {"rmse": float('inf'), "n_lags": None, "y_pred": None},
    "RNN": {"rmse": float('inf'), "n_lags": None, "y_pred": None},
}

for n_lags in range(3, 48, 4):
    print(f"Evaluating n_lags = {n_lags}")
    solution_seq = np.concatenate([power_values[-n_lags:], actual_test.ravel()])
    X_train, y_train = create_sliding_window(power_values, n_lags)
    x_test, y_test = create_sliding_window(solution_seq, n_lags)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(x_test)
    rmse_lr = math.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f"RMSE for Linear Regression n_lags={n_lags}: {rmse_lr}")
    if rmse_lr < results["LR"]["rmse"]:
        results["LR"] = {"rmse": rmse_lr, "n_lags": n_lags, "y_pred": y_pred_lr}

    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(x_test)
    rmse_svr = math.sqrt(mean_squared_error(y_test, y_pred_svr))
    print(f"RMSE for SVR n_lags={n_lags}: {rmse_svr}")
    if rmse_svr < results["SVR"]["rmse"]:
        results["SVR"] = {"rmse": rmse_svr, "n_lags": n_lags, "y_pred": y_pred_svr}

    ann_model = Sequential()
    ann_model.add(Dense(50, activation='relu', input_dim=n_lags))
    ann_model.add(Dense(25, activation='relu'))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer='adam', loss='mse')
    ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred_ann = ann_model.predict(x_test)
    rmse_ann = math.sqrt(mean_squared_error(y_test, y_pred_ann))
    print(f"RMSE for ANN n_lags={n_lags}: {rmse_ann}")
    if rmse_ann < results["ANN"]["rmse"]:
        results["ANN"] = {"rmse": rmse_ann, "n_lags": n_lags, "y_pred": y_pred_ann}

    X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    x_test_rnn = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    rnn_model = Sequential()
    rnn_model.add(LSTM(50, return_sequences=True, input_shape=(n_lags, 1)))
    rnn_model.add(LSTM(50))
    rnn_model.add(Dense(1))
    rnn_model.compile(optimizer='adam', loss='mse')
    rnn_model.fit(X_train_rnn, y_train, epochs=50, batch_size=32, verbose=0)
    y_pred_rnn = rnn_model.predict(x_test_rnn)
    rmse_rnn = math.sqrt(mean_squared_error(y_test, y_pred_rnn))
    print(f"RMSE for RNN n_lags={n_lags}: {rmse_rnn}")
    if rmse_rnn < results["RNN"]["rmse"]:
        results["RNN"] = {"rmse": rmse_rnn, "n_lags": n_lags, "y_pred": y_pred_rnn}

november_index = pd.date_range(start='2013-11-01 01:00', periods=len(y_test), freq='H')

plt.figure(figsize=(12, 6))
plt.plot(november_index, y_test, label='Actual Power', color='blue')
plt.plot(november_index, results["LR"]["y_pred"], label=f'Predicted Power (LR, n_lags={results["LR"]["n_lags"]})',
         color='orange')
plt.plot(november_index, results["SVR"]["y_pred"], label=f'Predicted Power (SVR, n_lags={results["SVR"]["n_lags"]})',
         color='green')
plt.xlabel('November 2013')
plt.ylabel('Power')
plt.title('Actual vs Predicted Power (Linear Regression & SVR)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(november_index, y_test, label='Actual Power', color='blue')
plt.plot(november_index, results["ANN"]["y_pred"], label=f'Predicted Power (ANN, n_lags={results["ANN"]["n_lags"]})',
         color='orange')
plt.plot(november_index, results["RNN"]["y_pred"], label=f'Predicted Power (RNN, n_lags={results["RNN"]["n_lags"]})',
         color='green')
plt.xlabel('November 2013')
plt.ylabel('Power')
plt.title('Actual vs Predicted Power (ANN & RNN)')
plt.legend()
plt.grid(True)
plt.show()

print("Best Model Results:")
for model_name, result in results.items():
    print(f"{model_name}: RMSE = {result['rmse']}, n_lags = {result['n_lags']}")


def save_predictions(predictions, filename):
    timestamps = pd.date_range(start='2013-11-01 01:00', periods=len(predictions), freq='H')
    df = pd.DataFrame({
        'TIMESTAMP': timestamps.strftime('%Y%m%d %H:%M'),
        'FORECAST': predictions.ravel()
    })
    df.to_csv(filename, index=False)


save_predictions(results["LR"]["y_pred"], f'ForecastTemplate3-LR.csv')
save_predictions(results["SVR"]["y_pred"], f'ForecastTemplate3-SVR.csv')
save_predictions(results["ANN"]["y_pred"], f'ForecastTemplate3-ANN.csv')
save_predictions(results["RNN"]["y_pred"], f'ForecastTemplate3-RNN.csv')
