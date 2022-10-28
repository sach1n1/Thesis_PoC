import importlib

import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import ProcessDB
import PredictSVR

importlib.reload(ProcessDB)
importlib.reload(PredictSVR)

import matplotlib.pyplot as plt


from ProcessDB import ProcessDB
from PredictSVR import PredictSVR


database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

forecast_hour = '2021-05-27 12:00:00'
training_duration = 6


df = ProcessDB(database_path, forecast_hour, training_duration)

df_predictions = PredictSVR(df.train, df.test)

plot_df = pd.DataFrame(df.test.iloc[-3601:])
plot_df = plot_df.rename(columns={'value': 'Test Values'})

plot_df["One Step Predictions"] = df_predictions.p_test[-3601:]
plot_df["Multi-Step Predictions"] = df_predictions.gp_test.reshape(1, -1)[0][-3601:]
plot_df.plot()
plt.xlabel("Time (seconds)")
plt.ylabel("Vibration Value")


plt.savefig(f'plots/SVR Multi-fail.jpg', format='jpg', dpi=1200)
plt.show()