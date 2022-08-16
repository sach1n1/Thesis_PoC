from copy import deepcopy

import pandas as pd
import warnings
from prophet import Prophet
from common.utils import load_data, mape, create_features, rmse
from time import time
from datetime import datetime
import sqlite3 as db
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"




database = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"
forecast_hour = '2021-05-27 12:00:00'
training_duration = 24
con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
                       f"time >= '{1622041200000}' AND time <= '{1622131200000}'",
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
#
train = train.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
test = test.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
print(test)

m = Prophet(changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.1)

m.fit(train)

future = m.make_future_dataframe(periods=len(test), freq='S', include_history=False)

forecast = m.predict(future)
print('MAPE for training data: ' + str(round(mape(forecast['yhat'], test['y'])*100, 2)) + '%' + "\n")
print('RMSE for training data: ' + str(round(rmse(forecast['yhat'], test['y']), 2)) + "\n")



pred = pd.DataFrame(index=test.index)
pred["Predicted Values"] = forecast['yhat']
pred["Test Values"] = test["y"]
print(pred)
pred.plot(title=f"Predictions based on {training_duration} hours of training data")
# plt.show()
# plt.savefig(f'Forecast with {training_duration} hours of training data.eps', format='eps', dpi=1200)


# iteration_dict["Train Start "] = str(train_start_dt)
# iteration_dict["Test Start "] = str(test_start_dt)
# iteration_dict["Test End"] = str(test_end_dt)
# iteration_dict["MAPE"] = round(mape(forecast['yhat'], test['y']) * 100, 2)
# iteration_dict["RMSE"] = round(rmse(forecast['yhat'], test['y']) * 100, 2)
# iteration_dict["Time Taken"] = round((time() - start) / 60, 2)



# result_dict[str(i)] = deepcopy(iteration_dict)
#
#
# train_start_dt = str(pd.Timestamp(train_start_dt) + pd.DateOffset(hours=1))
# test_start_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))
# test_end_dt = str(pd.Timestamp(test_start_dt) + pd.DateOffset(hours=1))

#    i = i + 1

# f = open("FBProphet/hour1.txt", 'wt')
# data = str(result_dict)
# f.write(data)
