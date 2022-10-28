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


vibration_x = load_data('../data')[['x']]
vibration = vibration_x.iloc[:5000]

test_len = int(len(vibration)*0.2)

train = vibration.iloc[:-test_len]
test = vibration.iloc[-test_len:]
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

multi_step = [x[0] for x in test_data[:timesteps-1]]
print(multi_step)
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

rmse_list = []
mape_list = []

for i in range(0, len(multi_step)-20):
    rmse_list.append(mean_squared_error(y_test[i:i+20], multi_step[i:i+20], squared=False))
    mape_list.append(mean_absolute_percentage_error(y_test[i:i+20], multi_step[i:i+20]) * 100)

plt.figure(figsize=(10, 5))
plt.plot(test_timestamps[:976], rmse_list, color='blue', linewidth=1.0, alpha = 0.6)
plt.axhline(y=0.045, color='r', linestyle='--')
# plt.title("RMSE Values for a rolling period of 20 timesteps")
plt.xlabel('Timesteps')
plt.ylabel('RMSE')
plt.savefig("plots/RMSE.jpg", format="jpg", dpi=1200)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(test_timestamps[:976], mape_list, color='blue', linewidth=1.0, alpha = 0.6)
plt.axhline(y=4, color='r', linestyle='--')
# plt.title("MAPE Values for a rolling period of 20 timesteps")
plt.xlabel('Timesteps')
plt.ylabel('MAPE (%)')
plt.savefig("plots/MAPE.jpg", format="jpg", dpi=1200)
plt.show()