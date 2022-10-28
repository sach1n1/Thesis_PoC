import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from time import time
import pandas as pd
import random
import datetime as dt
import math

from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy
from common.utils import load_data, mape, rmse, compar
import sqlite3 as db


vibration = load_data('/home/sachin/Thesis/data')[['vibration']]

train_start_dt = '2021-05-27 06:00:00'
test_start_dt = '2021-05-27 12:00:00'
test_end_dt = '2021-05-27 13:00:00'

ti = {
24:1622023200000,
12:1622066400000,
6:1622088000000,
1:1622106000000
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

#model = SVR(kernel='rbf', gamma=1, C=10, epsilon=0.01)
#model = SVR(kernel='rbf', gamma=1, C=10, epsilon=0.01)
model = SVR(kernel='rbf', gamma=0.01, C=10, epsilon=0.001)

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

print(f"MAPE= {round(mape(y_test_pred, y_test) * 100, 2)}")
print(f"RMSE= {round(rmse(y_test_pred, y_test), 2)}")

pred = pd.DataFrame(index=test_timestamps)
pred["Predicted Values"] = y_test_pred
pred["Test Values"] = y_test
print(pred)

pred.plot(title=f"Predictions based on {training_duration} hours of training data")
plt.show()
# plt.savefig(f'Forecast with {training_duration} hours of training data.eps', format='eps', dpi=1200)

# iteration_dict["Train Start "] = str(train_start_dt)
# iteration_dict["Test Start "] = str(test_start_dt)
# iteration_dict["Test End"] = str(test_end_dt)
# iteration_dict["MAPE"] = round(mape(y_test_pred, y_test) * 100, 2)
# iteration_dict["RMSE"] = round(rmse(y_test_pred, y_test), 2)
# iteration_dict["Time Taken"] = round((time() - start) / 60, 2)
# print(iteration_dict)

# print_date = test_start_dt

# mod_list = modifyList(y_test)

# for element in range(0, len(y_test_pred)):
#     # print(f"{print_date},{round(y_test_pred[element][0], 2)},{round(y_test[element][0], 2)}")
#     print_date = str(pd.Timestamp(print_date) + pd.DateOffset(seconds=1))

# result_dict[str(iterations)] = deepcopy(iteration_dict)

# train_start_dt = str(pd.Timestamp(train_start_dt) + pd.DateOffset(hours=1))
# test_start_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))
# test_end_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))

# f = open("SVR/hour6_gamma.txt", 'wt')
# data = str(result_dict)
# f.write(data)

# compar("SVR/hour6_ns.txt", "SVR/hour6.txt", "Time Taken")


# plt.figure(figsize=(25,6))
# plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
# plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
# plt.legend(['Actual','Predicted'])
# plt.xlabel('Timestamp')
# plt.title("Training data prediction")
# plt.show()

# print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')

# plt.figure(figsize=(10,3))
# plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
# plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
# plt.legend(['Actual','Predicted'])
# plt.xlabel('Timestamp')
# plt.show()
