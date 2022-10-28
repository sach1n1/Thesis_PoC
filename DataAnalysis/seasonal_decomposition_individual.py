import sqlite3 as db
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


plt.rcParams['agg.path.chunksize'] = 10000
database = "/home/sachin/Thesis/data/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=2 AND "
                       f"time >= '{1619136027000}' AND time < '{1619227253000}'",
                       con)
df["time"] = df["time"].apply(lambda utc: datetime.fromtimestamp(int(utc/1000)))
df.drop_duplicates(subset="time", keep="first", inplace=True)
df.index = df['time']
df.drop('time', axis=1, inplace=True)
df = df.reindex(pd.date_range(min(df.index),
                              max(df.index),
                              freq='S'))
df = df.interpolate().fillna(method='bfill')
con.close()
print(df)

res = seasonal_decompose(df["value"], model='additive', period=1)

plt.figure(figsize=(10, 6))
res.observed.plot()
plt.xticks(rotation=45)
plt.ylabel('Observed')
plt.xlabel("Time")
plt.savefig('plots/decomposition_observed.jpg', format='jpg', dpi=1200, bbox_inches='tight')

plt.figure(figsize=(10, 6))
res.trend.plot()
plt.xticks(rotation=45)
plt.ylabel('Trend')
plt.xlabel("Time")
plt.savefig('plots/decomposition_trend.jpg', format='jpg', dpi=1200, bbox_inches='tight')

plt.figure(figsize=(10, 6))
res.seasonal.plot()
plt.xticks(rotation=45)
plt.ylabel('Seasonality')
plt.xlabel("Time")
plt.savefig('plots/decomposition_season.jpg', format='jpg', dpi=1200, bbox_inches='tight')

plt.figure(figsize=(10, 6))
res.resid.plot()
plt.xticks(rotation=45)
plt.ylabel('Residual')
plt.xlabel("Time")
plt.savefig('plots/decomposition_resid.jpg', format='jpg', dpi=1200, bbox_inches='tight')
