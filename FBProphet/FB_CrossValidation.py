import itertools
import sys
from datetime import datetime
import pandas as pd
import warnings
import sqlite3 as db
import warnings
import numpy as np
from pandas import to_datetime, DataFrame
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from time import time

forecast_start_dt = '2021-06-21 11:00:00'

duration = 6

database_path = "../data/RWO_0004_Ventilatoren_00.sqlite"


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


train = train.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
test = test.reset_index().rename(columns={'index': 'ds', 'value': 'y'})

changepoint_prior_scale = [0.001, 0.01, 0.1, 1, 10, 100]
seasonality_prior_scale = [0.001, 0.01, 0.1, 1, 10, 100]


param_grid = {
    'changepoint_prior_scale': np.array(changepoint_prior_scale),
    'seasonality_prior_scale': np.array(seasonality_prior_scale),
    'n_changepoints' : [50,100,150,200]
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []


for params in all_params:
    m = Prophet(**params).fit(train)  # Fit model with given params
    df_cv = cross_validation(m, horizon='1 hours', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses

best_params = all_params[np.argmin(rmses)]
print(best_params)
