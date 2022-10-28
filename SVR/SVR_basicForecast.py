import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import sqlite3 as db


def load_data(data_dir):
    values = pd.read_csv(os.path.join(data_dir, 'Value1.csv')
                         , parse_dates=['timestamp'])
    values = values.drop_duplicates(subset="timestamp", keep='first')
    values.index = values['timestamp']
    values = values.reindex(pd.date_range(min(values['timestamp']),
                                          max(values['timestamp']),
                                          freq='S'))
    values = values.drop('timestamp', axis=1)
    values = values.interpolate()
    return values


vibration = load_data('/home/sachin/Thesis/data')[['vibration']]

train_start_dt = '2021-05-27 06:00:00'
test_start_dt = '2021-05-27 12:00:00'
test_end_dt = '2021-05-27 13:00:00'

ti = {
    24: 1622023200000,
    12: 1622066400000,
    6: 1622088000000,
    1: 1622106000000
}

scaler = MinMaxScaler()

database = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"
forecast_hour = '2021-05-27 12:00:00'
training_duration = 1
con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
                       f"time >= '{1622023200000}' AND time <= '{1622131201000}'",
                       con)
df["time"] = df["time"].apply(lambda utc: datetime.fromtimestamp(int(utc / 1000)))
df.drop_duplicates(subset="time", keep="first", inplace=True)
df.index = df['time']
df = df.reindex(pd.date_range(min(df.index),
                              max(df.index),
                              freq='S'))
df.drop('time', axis=1, inplace=True)
df = df.interpolate().fillna(method='bfill')
con.close()

test_len = 3600
train, test = df.iloc[:-test_len], df.iloc[-test_len:]

train['value'] = scaler.fit_transform(train)

test['value'] = scaler.transform(test)

train_data = train.values
test_data = test.values

timesteps = 5

train_data_timesteps = np.array(
    [[j for j in train_data[i:i + timesteps]] for i in range(0, len(train_data) - timesteps + 1)])[:, :, 0]

test_data_timesteps = np.array(
    [[j for j in test_data[i:i + timesteps]] for i in range(0, len(test_data) - timesteps + 1)])[:, :, 0]

x_train, y_train = train_data_timesteps[:, :timesteps - 1], train_data_timesteps[:, [timesteps - 1]]
x_test, y_test = test_data_timesteps[:, :timesteps - 1], test_data_timesteps[:, [timesteps - 1]]


#model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
model = SVR(kernel='rbf', gamma=0.01, C=10, epsilon=0.001) #CV

model.fit(x_train, y_train[:, 0])

y_train_pred = model.predict(x_train).reshape(-1, 1)
y_test_pred = model.predict(x_test).reshape(-1, 1)

y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

train_timestamps = vibration[(vibration.index < test_start_dt) & (vibration.index >= train_start_dt)].index[
                   timesteps - 1:]
test_timestamps = vibration[(vibration.index < test_end_dt) & (vibration.index >= test_start_dt)].index[
                  timesteps - 1:]

print(f"MAPE= {round(mean_absolute_percentage_error(y_test_pred, y_test) * 100, 2)}")
print(f"RMSE= {round(mean_squared_error(y_test_pred, y_test, squared=False), 2)}")

pred = pd.DataFrame(index=test_timestamps)
pred["Predicted Values"] = y_test_pred
pred["Test Values"] = y_test
print(pred)

pred.plot(title=f"Predictions based on {training_duration} hours of training data")
plt.show()
plt.savefig(f'Forecast with {training_duration} hours of training data.jpg', format='jpg', dpi=1200)