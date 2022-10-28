import sqlite3 as db
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.graphics.tsaplots as smt


def tsplot(y, lags=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    smt.plot_acf(y, lags=lags)


plt.rcParams['agg.path.chunksize'] = 10000
database = "../data/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=2 AND "
                       f"time >= '{1619136027000}' AND time < '{1619289641000}'",
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

plt.acorr(df['value'], normed=True, maxlags=100, lw=2)
plt.xlim([0, 100])
plt.xlabel("Lags")
plt.ylabel("Autocorrelation")
plt.savefig('plots/acf_pacf.jpg', format='jpg', dpi=1200)