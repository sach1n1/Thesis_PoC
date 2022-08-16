from copy import deepcopy

import pandas as pd
import warnings
from prophet import Prophet
from common.utils import load_data, mape, create_features, rmse
from time import time
from sklearn.metrics import mean_squared_error
from SVR_opt.ProcessDB import  ProcessDB
import matplotlib.pyplot as plt
import sqlite3 as db
from datetime import datetime

warnings.simplefilter('ignore')


database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

forecast_hour = '2021-05-27 12:00:00'
training_duration = 6


database = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
                       f"time >= '{1619820000000}' AND time < '{1619848800000}'",
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




df = df[:1000]

test_len = int(len(df) * 0.2)
train, test = df.iloc[:-test_len], df.iloc[-test_len:]

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#from statsmodels.tsa.api import SimpleExpSmoothing

ses = SimpleExpSmoothing(train)

alpha = 0.0001
model = ses.fit(smoothing_level=alpha, optimized = False)

forcast = model.forecast(len(test))

print(forcast)
test["forecast"] = forcast
test.plot()
plt.show()
# train = train.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
# test = test.reset_index().rename(columns={'index': 'ds', 'value': 'y'})


# test = test.set_index('ds')
# print(test['ds'])
#test = test.value.resample('1min').mean()
# print(test)

# m = Prophet(growth="linear",
#             changepoint_prior_scale=1.0,
#             daily_seasonality=True,
#             seasonality_prior_scale=0.001,
#             n_changepoints=100)
#
# m.fit(train)
#
# future = m.make_future_dataframe(periods=len(test), freq='S', include_history=False)
# #
# forecast = m.predict(future)
# print(forecast)
# print(f"rmse:{mean_squared_error(forecast['yhat'], test['y'], squared=False)}")
# print(f"mape:{mape(forecast['yhat'], test['y'])}")
# lm = pd.DataFrame(index=forecast.index)
# lm['yhat'] = forecast['yhat']
# lm['y'] = test['y']
# print(lm)
# plt.plot(lm)
# plt.show()
#
# # iteration_dict["Train Start "] = str(train_start_dt)
# iteration_dict["Test Start "] = str(test_start_dt)
# iteration_dict["Test End"] = str(test_end_dt)
# iteration_dict["MAPE"] = round(mape(forecast['yhat'], test['y']) * 100, 2)
# iteration_dict["RMSE"] = round(rmse(forecast['yhat'], test['y']) * 100, 2)
# iteration_dict["Time Taken"] = round((time() - start) / 60, 2)
#
# result_dict[str(i)] = deepcopy(iteration_dict)
#
#
# train_start_dt = str(pd.Timestamp(train_start_dt) + pd.DateOffset(hours=1))
# test_start_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))
# test_end_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))

