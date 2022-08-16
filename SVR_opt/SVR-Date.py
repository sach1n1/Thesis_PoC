from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

from common.utils import load_data, mape, rmse, compar


vibration = load_data('/home/sachin/Thesis/data')[['vibration']]



train_start_dt = '2021-05-27 04:00:00'
test_start_dt = '2021-05-27 10:00:00'
test_end_dt = '2021-05-27 11:00:00'

scaler = MinMaxScaler()

result_dict = dict()
iteration_dict = dict()


train = vibration.copy()[(vibration.index >= train_start_dt) & (vibration.index < test_start_dt)][['vibration']]
test = vibration.copy()[(vibration.index >= test_start_dt) & (vibration.index < test_end_dt)][['vibration']]


from sklearn.svm import SVR

train['vibration'] = scaler.fit_transform(train)

test['vibration'] = scaler.transform(test)

train_data = train["vibration"].tolist()
test_data = test.values


import numpy as np

train_dates = [int(x.timestamp()) for x in train.index]
test_dates = [int(x.timestamp()) for x in test.index]
train_dates = np.array(train_dates)
test_dates = np.array(test_dates)
train_dates = np.reshape(train_dates, (len(train_dates), 1))
test_dates = np.reshape(test_dates, (len(test_dates), 1))

print("train start", datetime.now())


model = SVR(kernel='sigmoid', gamma=0.001, C=1, epsilon=0.001)

model.fit(train_dates, train_data)
print("train end", datetime.now())
y_test_pred = model.predict(test_dates).reshape(-1,1)
test['vibration'] = scaler.inverse_transform(test)
y_test_pred = scaler.inverse_transform(y_test_pred)
print(y_test_pred)
print(test.values)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(test.values,y_test_pred,squared=False))
print(mape(y_test_pred, test.values))


