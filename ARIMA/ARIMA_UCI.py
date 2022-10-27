

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd



database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

forecast_hour = '2021-05-27 12:00:00'
training_duration = 6

scaler = MinMaxScaler()

database = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
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


scaler = MinMaxScaler()

df.train['value'] = scaler.fit_transform(df.train)
df.test['value'] = scaler.transform(df.test)

HORIZON = 904

order = (4, 1, 0)


model = SARIMAX(endog=df.train, order=order, seasonal_order=seasonal_order)
results = model.fit()

print(results.summary())



yhat = results.forecast(steps = HORIZON)



yhat = scaler.inverse_transform(yhat.values.reshape(-1,1))
print(yhat)
