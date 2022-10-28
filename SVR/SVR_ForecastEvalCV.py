import numpy as np
from time import time
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


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
    iteration_dict["MAPE"] = round(mean_absolute_percentage_error(y_test_pred, y_test) * 100, 2)
    iteration_dict["RMSE"] = round(mean_squared_error(y_test_pred, y_test, squared=False), 2)
    iteration_dict["Time Taken"] = round((time() - start) / 60, 2)

    result_dict[str(iterations)] = deepcopy(iteration_dict)

    train_start_dt = str(pd.Timestamp(train_start_dt) + pd.DateOffset(hours=1))
    test_start_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))
    test_end_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))

f = open("Outs/hour24CV.txt", 'w')
data = str(result_dict)
print(data)
f.write(data)
f.close()
