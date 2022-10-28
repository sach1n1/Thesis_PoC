import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from time import time
import pandas as pd
import datetime as dt
import math

from datetime import datetime

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn.model_selection import KFold
from common.utils import load_data, mape, rmse



start = time()

vibration = load_data('/home/sachin/Thesis/data')[['vibration']]

train_start_dt = '2021-05-27 11:00:00'
test_start_dt = '2021-05-27 12:00:00'
test_end_dt = '2021-05-27 13:00:00'

scaler = MinMaxScaler()

result_dict = dict()
iteration_dict = dict()

# for iterations in range(1, 31):
#     print(iterations)
start = time()

train = vibration.copy()[(vibration.index >= train_start_dt) & (vibration.index < test_start_dt)][['vibration']]
test = vibration.copy()[(vibration.index >= test_start_dt) & (vibration.index < test_end_dt)][['vibration']]

train['vibration'] = scaler.fit_transform(train)

test['vibration'] = scaler.transform(test)

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
model = SVR(kernel='rbf', gamma=0.1, C=10, epsilon=0.01) #CV

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

iteration_dict["Train Start "] = str(train_start_dt)
iteration_dict["Test Start "] = str(test_start_dt)
iteration_dict["Test End"] = str(test_end_dt)
iteration_dict["MAPE"] = round(mape(y_test_pred, y_test) * 100, 2)
iteration_dict["RMSE"] = round(rmse(y_test_pred, y_test), 2)
iteration_dict["Time Taken"] = round((time() - start) / 60, 2)
print(f'MAPE: {iteration_dict["MAPE"]}')
print(f'RMSE: {iteration_dict["RMSE"]}')

# plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color='red')
plt.plot(test_timestamps, y_test_pred, color='blue')
plt.legend(['Actual', 'Predicted'])
plt.ylabel("Vibration Value")
plt.xlabel("Time (seconds)")
plt.savefig(f"plots/SVR1_CV.jpg", format="jpg", dpi=1200)
plt.show()
