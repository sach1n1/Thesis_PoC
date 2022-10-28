import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

warnings.simplefilter('ignore')


def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'accelerometer.csv'))
    return energy

vibration_x = load_data('../data/')[['x']]

df = vibration_x.iloc[:5000]


df = df[:5000]

test_len = int(len(df) * 0.2)
train, test = df.iloc[:-test_len], df.iloc[-test_len:]

forecast = test.shift(1)

plt.figure(figsize=(10, 4))
plt.plot(test, color='red', linewidth=2.0, alpha=0.6)
plt.plot(forecast, color='green', linewidth=2.0)
plt.legend(['Actual Value', 'Predictions'], loc="upper right")
plt.xlabel('Timesteps')
plt.ylabel('Acceleration (g)')
plt.savefig("plots/naive_huge_prediction.jpg", format="jpg", dpi=1200)

plt.plot(forecast)
plt.plot(test)
plt.show()

forecast = forecast['x'].fillna(0)

print(mean_squared_error(test, forecast, squared=False))
print(mean_absolute_percentage_error(test, forecast)*100)
