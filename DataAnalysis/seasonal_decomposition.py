import sqlite3 as db
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.graphics.tsaplots as smt
from statsmodels.tsa.seasonal import seasonal_decompose

def plotseasonal(res):
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')

# def tsplot(y, lags=None, figsize=(12, 7)):
#     if not isinstance(y, pd.Series):
#         y = pd.Series(y)
#
#     fig = plt.figure(figsize=figsize)
#     layout = (2, 1)
#     #ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
#     acf_ax = plt.subplot2grid(layout, (0, 0))
#     pacf_ax = plt.subplot2grid(layout, (1, 0))
#
#     #y.plot(ax=ts_ax)
#     #p_value = sm.tsa.stattools.adfuller(y)[1]
#     #ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
#     smt.plot_acf(y, lags=lags, ax=acf_ax)
#     smt.plot_pacf(y, lags=lags, ax=pacf_ax)
#     plt.tight_layout()


def tsplot(y, lags=None):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    smt.plot_acf(y, lags=lags)


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
print(df)

res = seasonal_decompose(df["value"], model='additive', period=1)

fig, axes = plt.subplots(nrows=4, sharex=True)
fig.suptitle('Decomposition of Vibration Values into components')
plotseasonal(res)

plt.tight_layout(pad=1.08, rect=[0, 0.03, 1, 0.95])

# result.plot()
plt.xticks(rotation=45)
plt.xlabel("Time")
plt.savefig('plots/decomposition.eps', format='eps', dpi=1200)
plt.savefig('plots/decomposition.jpg', format='jpg', dpi=1200)

plt.show()


tsplot(df['value'])
plt.xlabel("Lags")
plt.ylabel("Correlation Factor")
plt.title("Autocorrelation Plot")
plt.savefig('plots/acf_pacf.eps', format='eps', dpi=1200)
plt.savefig('plots/acf_pacf.jpg', format='jpg', dpi=1200)
plt.show()