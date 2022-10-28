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

train_start_dt = '2021-05-26 12:00:00'
test_start_dt = '2021-05-27 12:00:00'
test_end_dt = '2021-05-27 13:00:00'

scaler = MinMaxScaler()

result_dict = dict()
iteration_dict = dict()

for iterations in range(1, 31):
    print(iterations)
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

    # model = SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05)
    # model = SVR(kernel='rbf', gamma=1, C=10, epsilon=0.01) #CV take slongerthesis
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

    iteration_dict["Train Start "] = str(train_start_dt)
    iteration_dict["Test Start "] = str(test_start_dt)
    iteration_dict["Test End"] = str(test_end_dt)
    iteration_dict["MAPE"] = round(mape(y_test_pred, y_test) * 100, 2)
    iteration_dict["RMSE"] = round(rmse(y_test_pred, y_test), 2)
    iteration_dict["Time Taken"] = round((time() - start) / 60, 2)

    result_dict[str(iterations)] = deepcopy(iteration_dict)

    train_start_dt = str(pd.Timestamp(train_start_dt) + pd.DateOffset(hours=1))
    test_start_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))
    test_end_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))



f = open("Outs/hour24CVp.txt", 'w')
data = str(result_dict)
print(data)
f.write(data)
f.close()

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
