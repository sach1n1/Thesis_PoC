import sqlite3 as db
import pandas as pd
import matplotlib.pyplot as plt
database = "/home/sachin/Thesis/data/RWO_0004_Ventilatoren_00.sqlite"

con = db.connect(database)
df = pd.read_sql_query(f"SELECT time, value FROM Value WHERE sensor_id=1",
                       con)
df.index = df['time']
df.drop('time', axis=1, inplace=True)
con.close()
print(df)

ax = df.plot(figsize=(10, 6))
ax.legend(['Vibration Values'])
plt.title('Time Series Vibration Dataset')
plt.xlabel('Timesteps (s)')
plt.ylabel('Acceleration (g)')
plt.savefig('plots/full_data_unixds.eps', format='eps', dpi=1200)
plt.savefig('plots/full_data_unixds.jpg', format='jpg', dpi=1200)
plt.show()
