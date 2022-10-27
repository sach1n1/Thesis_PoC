import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import sqlite3 as db
from datetime import datetime
from ProcessDB import ProcessDB
from PredictSVR import PredictSVR
import numpy as np
from sklearn.svm import SVR

database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

forecast_hour = '2021-05-27 12:00:00'
training_duration = 6


def load_required_value():
    train_start_dt = str(pd.Timestamp(forecast_hour) - pd.DateOffset(hours=training_duration))
    forecast_end_dt = str(pd.Timestamp(forecast_hour) + pd.DateOffset(minutes=15))
    df_train = create_data_frame(train_start_dt, forecast_hour)
    df_test = create_data_frame(forecast_hour, forecast_end_dt)
    return df_train, df_test

def create_data_frame(start_date, end_date):
    con = db.connect(database_path)
    start_dt_utc = datetime.timestamp(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')) * 1000
    end_dt_utc = datetime.timestamp(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')) * 1000
    df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=2 AND "
                           f"time >= '{int(start_dt_utc)}' AND time < '{int(end_dt_utc)}'",
                           con)
    df["time"] = df["time"].apply(lambda utc: datetime.fromtimestamp(int(utc / 1000)))
    df.drop_duplicates(subset="time", keep="first", inplace=True)
    df.index = df['time']
    df = df.reindex(pd.date_range(start_date,
                                  end_date,
                                  freq='S'))
    df.drop('time', axis=1, inplace=True)
    df = df.interpolate().fillna(method='bfill')
    df.drop(df.tail(1).index, inplace=True)
    con.close()
    print(df)
    return df


train, test = load_required_value()

timesteps = 5

scaler = StandardScaler()
train['value'] = scaler.fit_transform(train)
test['value'] = scaler.transform(test)

train_data = train.values
test_data = test.values

train_data_timesteps = np.array(
    [[j for j in train_data[i:i + timesteps]] for i in range(0, len(train_data) - timesteps + 1)])[:,
                       :, 0]

test_data_timesteps = np.array(
    [[j for j in test_data[i:i + timesteps]] for i in range(0, len(test_data) - timesteps + 1)])[:, :,
                      0]

x_train, y_train = train_data_timesteps[:, :timesteps - 1], train_data_timesteps[:, [timesteps - 1]]
x_test, y_test = test_data_timesteps[:, :timesteps - 1], test_data_timesteps[:, [timesteps - 1]]

model = SVR(kernel='rbf', gamma=0.01, C=100, epsilon=0.001)

model.fit(x_train, y_train[:, 0])

y_train_pred = model.predict(x_train).reshape(-1, 1)
y_test_pred = model.predict(x_test).reshape(-1, 1)

y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

y_mov_test = [x[0] for x in test_data[:100]]
print(y_mov_test)
# for i in range(0, len(test)):
#     pred = model.predict([[y_mov_test[-9], y_mov_test[-8], y_mov_test[-7], y_mov_test[-6], y_mov_test[-5],
#                            y_mov_test[-4], y_mov_test[-3], y_mov_test[-2], y_mov_test[-1]]]).reshape(-1, 1)
#     y_mov_test.append(pred[0][0])
# y_mov_test = y_mov_test[-len(self.test):]
# y_mov_test = np.array(y_mov_test).reshape(-1, 1)
#
# y_mov_test = scaler.inverse_transform(y_mov_test)
#
# df_predictions = PredictSVR(df.train, df.test)
#
# # print("Test Predictions")
# # print(mean_squared_error(df.test[-895:], df_predictions.p_test))
# #
# #
# # print("Real Predictions")
# # print(mean_squared_error(df.test[-895:], df_predictions.gp_test[-895:]))
#
#
# import matplotlib.pyplot as plt
# pred = df_predictions.gp_test[-895:]
# pred = pred.reshape(1,-1)
# new_df = df.test[-895:]
# new_df["predict"] = pred[0]
# new_df["test"] = df_predictions.p_test
# plt.plot(new_df)
# plt.show()