import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def load_data(data_dir):
    energy = pd.read_csv(os.path.join(data_dir, 'accelerometer.csv'))
    return energy


vibration_x = load_data('/home/sachin/Downloads')[['x']]

print(len(vibration_x))

vibration_x.plot()

vibration_x.hist(bins=30)
plt.title("Distribution of Acceleration for UCI Repository Dataset")
plt.xlabel('Acceleration: X-axis (g)')
plt.savefig("hist_vib.eps", format="eps", dpi=1200)
plt.savefig("hist_vib.jpg", format="jpg", dpi=1200)
plt.show()

vibration = vibration_x.iloc[:5000]
vibration.plot()
plt.show()

test_len = int(len(vibration)*0.2)

train = vibration.iloc[:-test_len]
test = vibration.iloc[-test_len:]
scaler = StandardScaler()
train['x'] = scaler.fit_transform(train)
test['x'] = scaler.transform(test)

train_data = train.values
test_data = test.values

timesteps = 10


train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]

test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]

x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

model = SVR(kernel='rbf', gamma=1, C=70, epsilon = 0.0001)

model.fit(x_train, y_train[:,0])

y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

multi_step = [x[0] for x in test_data[:timesteps-1]]
print(multi_step)
for i in range(0, len(y_test_pred)):
    pred_i = model.predict([multi_step[-(timesteps-1):]])
    # pred_i = model.predict([[y_mov_test[-9], y_mov_test[-8], y_mov_test[-7], y_mov_test[-6], y_mov_test[-5],
    #                          y_mov_test[-4], y_mov_test[-3], y_mov_test[-2], y_mov_test[-1]]]).reshape(-1, 1)
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
# plt.plot(test_timestamps, y_test_pred, color='blue', linewidth=2.0)
plt.plot(test_timestamps, multi_step, color='green', linewidth=2.0)
plt.legend(['Actual Value', 'Multi-Step Predictions'], loc="upper right")
# plt.legend(['Actual Value', 'Single-Step Predictions', 'Multi-Step Predictions'])
plt.title("Actual Values vs Multi-Step Predictions")
plt.xlabel('Timesteps (milliseconds)')
plt.ylabel('Acceleration: X-axis (g)')
print(mean_squared_error(y_test, multi_step, squared=False))
print(mean_absolute_percentage_error(y_test, multi_step)*100)


plt.savefig("Vibration-x_huge_prediction.eps", format="eps", dpi=1200)
plt.savefig("Vibration-x_huge_prediction.jpg", format="jpg", dpi=1200)

plt.show()
