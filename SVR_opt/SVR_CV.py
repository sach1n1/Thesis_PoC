
import numpy as np
from time import time
import sqlite3 as db
from datetime import datetime
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

start = time()


forecast_start_dt = '2021-06-21 11:00:00'

duration = 6

database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"


def load_required_value(forecast_start_dt):
    train_start_dt = str(pd.Timestamp(forecast_start_dt) - pd.DateOffset(hours=duration, seconds=4))
    forecast_end_dt = str(pd.Timestamp(forecast_start_dt) + pd.DateOffset(hours=duration))
    df_train = create_data_frame(train_start_dt, forecast_start_dt)
    forecast_start_dt = str(pd.Timestamp(forecast_start_dt) - pd.DateOffset(seconds=4))
    df_test = create_data_frame(forecast_start_dt, forecast_end_dt)
    return df_train, df_test

def create_data_frame(start_date, end_date):
    con = db.connect(database_path)
    start_dt_utc = datetime.timestamp(datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')) * 1000
    end_dt_utc = datetime.timestamp(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')) * 1000
    df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
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
    return df


train, test = load_required_value(forecast_start_dt)

scaler = MinMaxScaler()

train['value'] = scaler.fit_transform(train)

test['value'] = scaler.transform(test)


train_data = train.values
test_data = test.values

timesteps=5

train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0, len(train_data)-timesteps+1)])[ :, :, 0]

test_data_timesteps = np.array([[j for j in test_data[i:i+timesteps]] for i in range(0, len(test_data)-timesteps+1)])[:, :, 0]

X_train, y_train = train_data_timesteps[:, :timesteps-1], train_data_timesteps[:, [timesteps-1]]
X_test, y_test = test_data_timesteps[:, :timesteps-1], test_data_timesteps[:, [timesteps-1]]


params = {
    'kernel': ['rbf', 'linear', 'sigmoid'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1,  1, 10, 100]
    }

grid_search = GridSearchCV(SVR(), params, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train[:, 0])

print("train score - " + str(grid_search.score(X_train, y_train)))
print("test score - " + str(grid_search.score(X_test, y_test)))

print("SVR GridSearch score: "+str(grid_search.best_score_))
print("SVR GridSearch params: ")
print(grid_search.best_params_)