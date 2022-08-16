from copy import deepcopy

import pandas as pd
import warnings
from prophet import Prophet
from common.utils import load_data, mape, create_features, rmse
from time import time
from sklearn.metrics import mean_squared_error
from SVR_opt.ProcessDB import ProcessDB
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sqlite3 as db
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

warnings.simplefilter('ignore')


database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

forecast_hour = '2021-05-27 12:00:00'
training_duration = 6

database = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
#                       f"time >= '{1619820000000}' AND time < '{1622412000000}'",
                       f"time >= '{1622412000000}' AND time < '{1622498400000}'",
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
plt.plot(df)
plt.show()

df = df.diff(5)
df.dropna(axis=0, inplace=True)

df = df[:1000]

train = df.iloc[:-200]
test = df.iloc[-200:]
scaler = StandardScaler()
train['x'] = scaler.fit_transform(train)
test['x'] = scaler.transform(test)

train_data = train.values
test_data = test.values

timesteps = 10


train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]

test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]

x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon = 0.05)

model.fit(x_train, y_train[:,0])

y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

multi_step = [x[0] for x in test_data[:timesteps-1]]
print(multi_step)
for i in range(0, len(y_test_pred)):
    pred_i = model.predict([multi_step[-(timesteps-1):]])
    # pred_i = model.predict([[y_mov_test[-9], y_mov_test[-8], y_mov_test[-7], y_mov_test[-6], y_mov_test[-5],
    #                          y_mov_test[-4], y_mov_test[-3], y_mov_test[-2], y_mov_test[-1]]]).reshape(-1, 1)
    multi_step.append(pred_i[0])
multi_step = multi_step[-len(y_test_pred):]


y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)
multi_step = scaler.inverse_transform(multi_step)

train_timestamps = list(train.index)[timesteps-1:]
test_timestamps = list(test.index)[timesteps-1:]


plt.figure(figsize=(10,4))
plt.plot(test_timestamps[:60], y_test[:60], color='red', linewidth=2.0, alpha = 0.6)
# plt.plot(test_timestamps, y_test_pred, color='blue', linewidth=2.0)
plt.plot(test_timestamps[:60], multi_step[:60], color='green', linewidth=2.0)
plt.legend(['Actual Value', 'Multi-Step Predictions'], loc="upper right")
#plt.legend(['Actual Value', 'Single-Step Predictions', 'Multi-Step Predictions'])
plt.title("Actual Values vs Multi-Step Predictions")
plt.xlabel('Timesteps (milliseconds)')
plt.ylabel('Acceleration: X-axis (g)')


# df = df[:1000]
# df = df.diff(2)
#
# scaler = MinMaxScaler()
#
# df["value"] = scaler.fit_transform(df)
#
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#
# fig, ax = plt.subplots(2, figsize=(12,6))
# ax[0] = plot_acf(df.dropna(), ax=ax[0], lags=20)
# ax[1] = plot_pacf(df.dropna(), ax=ax[1], lags=20)
plt.show()