import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'accelerometer.csv'))
    return energy


vibration_x = load_data('../data/')[['x']]

vibration_x.iloc[:5000].plot(legend=None)
plt.xlabel("Timesteps")
plt.ylabel("Acceleration (g)")
plt.savefig("plots/Vibration-x.jpg", format="jpg", dpi=1200)


vibration = vibration_x.iloc[:1000]

train = vibration.iloc[:-200]
test = vibration.iloc[-200:]
scaler = StandardScaler()
train['x'] = scaler.fit_transform(train)
test['x'] = scaler.transform(test)

train_data = train.values
test_data = test.values

timesteps = 5


train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]

test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]

x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

model = SVR(kernel='rbf', gamma=1, C=70, epsilon = 0.0001)

model.fit(x_train, y_train[:,0])

y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

multi_step = [x[0] for x in test_data[:(timesteps-1)]]

for i in range(0, len(y_test_pred)):
    pred_i = model.predict([multi_step[-(timesteps-1):]])
    multi_step.append(pred_i[0])
multi_step = multi_step[-len(y_test_pred):]


y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)
multi_step = scaler.inverse_transform(multi_step)

train_timestamps = list(train.index)[timesteps-1:]
test_timestamps = list(test.index)[timesteps-1:]

plt.figure(figsize=(10, 4))
plt.plot(test_timestamps, y_test, color='red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color='blue', linewidth=2.0)
plt.legend(['Actual Value', 'Single-Step Predictions'], loc="upper right")
plt.xlabel('Timesteps')
plt.ylabel('Acceleration (g)')
plt.savefig("plots/UCI-short-one.jpg", format="jpg", dpi=1200)
print("One-Step")
print(mean_squared_error(y_test,  y_test_pred, squared=False))
print(mean_absolute_percentage_error(y_test,  y_test_pred)*100)
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(test_timestamps, y_test, color='red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, multi_step, color='green', linewidth=2.0)
plt.legend(['Actual Value', 'Multi-Step Predictions'], loc="upper right")
plt.xlabel('Timesteps')
plt.ylabel('Acceleration (g)')
plt.savefig("plots/UCI-short-multi.jpg", format="jpg", dpi=1200)
print("Multi-Step")
print(mean_squared_error(y_test,  multi_step, squared=False))
print(mean_absolute_percentage_error(y_test,  multi_step)*100)
plt.show()