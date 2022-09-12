import sqlite3 as db
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm

plt.rcParams['agg.path.chunksize'] = 10000
database = "/home/sachin/Thesis/data/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=2 AND "
                       f"time >= '{1619136027000}' AND time < '{1648764000000}'",
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

result = sm.tsa.stattools.adfuller(df["value"])

print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

if result[1] < 0.05 and result[0] < result[4]['5%']:
    print("Stationary")
else:
    print("Non Stationary")