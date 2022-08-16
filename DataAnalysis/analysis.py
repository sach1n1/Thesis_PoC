import sqlite3 as db

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
database = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1 AND "
                       f"time >= '{1619820000000}' AND time < '{1622498400000}'",
                       con)
df["time"] = df["time"].apply(lambda utc: datetime.fromtimestamp(int(utc / 1000)))
df.drop_duplicates(subset="time", keep="first", inplace=True)
df.index = df['time']
df = df.reindex(pd.date_range(min(df.index),
                              max(df.index),
                              freq='s'))
df.drop('time', axis=1, inplace=True)
df = df.interpolate().fillna(method='bfill')
con.close()

df = df.resample('1440min').mean()


y = df["value"]
y.name = "Vibration Value"
#y.name = "n_passengers"
print(y)

seasonal_decomp = seasonal_decompose(y, model="additive", period=1)
seasonal_decomp.plot()
plt.show()

