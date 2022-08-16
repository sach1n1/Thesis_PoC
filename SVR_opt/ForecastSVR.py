import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from ProcessDB import ProcessDB
from PredictSVR import PredictSVR
import matplotlib.pyplot as plt


database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

forecast_hour = '2021-05-27 12:00:00'
training_duration = 6


df = ProcessDB(database_path, forecast_hour, training_duration)

df_predictions = PredictSVR(df.train, df.test)

# print("Test Predictions")
# print(mean_squared_error(df.test[-895:], df_predictions.p_test))
#
#
# print("Real Predictions")
# print(mean_squared_error(df.test[-895:], df_predictions.gp_test[-895:]))
#print(df_predictions.test)

# import matplotlib.pyplot as plt
#print(df_predictions.gp_test.reshape(1, -1))
plot_df = pd.DataFrame(df.test.iloc[-3600:])
print(plot_df)
plot_df = plot_df.rename(columns={'value': 'Test Values'})
#plot_df["Test Values"] = df_predictions.test.iloc[-3600:]
plot_df["One Step Predictions"] = df_predictions.p_test[-3600:]
plot_df["Multi-Step Predictions"] = df_predictions.gp_test.reshape(1, -1)[0][-3600:]
print(plot_df.head())
plt.xlabel("Time")
plt.xlabel("Vibration Value")
plot_df.plot(title="SVR Test Value vs One Step and Multi Step Prediction")

# plt.savefig(f'SVR Multi-fail.eps', format='eps', dpi=1200)
# plt.savefig(f'SVR Multi-fail.jpg', format='jpg', dpi=1200)
plt.show()
# new_df = df.iloc[-3600:]
# new_df["predict"] = pred[0]
#
# new_df["test"] = df_predictions.p_test
# plt.plot(new_df)
# plt.show()
