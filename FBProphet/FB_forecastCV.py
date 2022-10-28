import pandas as pd
import warnings
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
import sqlite3 as db
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

database = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"
forecast_hour = '2021-05-27 12:00:00'

ti = {
24:1622023200000,
12:1622066400000,
6:1622088000000,
1:1622106000000
}

training_duration = 24
con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
                       f"time >= '{ti[24]}' AND time <= '{1622113201000}'",
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


test_len = 3601
train, test = df.iloc[:-test_len], df.iloc[-test_len:]
#
train = train.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
test = test.reset_index().rename(columns={'index': 'ds', 'value': 'y'})
print(test)


m = Prophet(growth="linear",
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.001,
            n_changepoints=100)

m.fit(train)

future = m.make_future_dataframe(periods=len(test), freq='S', include_history=False)

forecast = m.predict(future)
print('MAPE for training data: ' + str(round(mean_absolute_percentage_error(forecast['yhat'], test['y'])*100, 2)) + '%' + "\n")
print('RMSE for training data: ' + str(round(mean_squared_error(forecast['yhat'], test['y']), 2)) + "\n")


pred = pd.DataFrame(index=test.index)
pred["Predicted Values"] = forecast['yhat']
pred["Test Values"] = test["y"]
pred.index = test['ds']
print(pred)
pred.plot()
plt.xlabel("Time (seconds)")
plt.ylabel("Vibration Value")
plt.savefig(f'plots/Forecast with {training_duration} hours of training data_cv.jpg', format='jpg', dpi=1200)