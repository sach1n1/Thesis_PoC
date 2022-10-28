import os
import sqlite3 as db
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'accelerometer.csv'))
    return energy

bin=30

vibration_x = load_data('../data')[['x']]

vibration_x.hist(bins=bin)
plt.title("")
# plt.title("Distribution of Acceleration for UCI Repository Dataset")
plt.xlabel('Acceleration Values')
plt.savefig(f"plots/hist_vib_UCI_{bin}.jpg", format="jpg", dpi=1200)
plt.show()

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

df["value"].hist(bins=bin)
plt.title("")
# plt.title("Azeti Dataset")
plt.xlabel('Vibration Values')
plt.savefig(f"plots/hist_vib_azeti_{bin}.jpg", format="jpg", dpi=1200)
plt.show()