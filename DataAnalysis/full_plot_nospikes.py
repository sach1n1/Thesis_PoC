import sqlite3 as db
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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
# df = df.interpolate().fillna(method='bfill')
con.close()
print(df)

ax = df.plot(figsize=(10, 6))
ax.legend(['Vibration Values'])
plt.title('Time Series Vibration Dataset')
plt.xlabel('Timesteps (s)')
plt.ylabel('Acceleration (g)')
plt.savefig('plots/full_data_no_spikes_noipl.eps', format='eps', dpi=1200)
plt.savefig('plots/full_data_no_spikes_noipl.jpg', format='jpg', dpi=1200)
plt.show()
