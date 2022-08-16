
import numpy as np
from time import time


from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from common.utils import load_data

start = time()

energy = load_data('../data')[['vibration']]


train_start_dt = '2021-06-21 07:00:00'
test_start_dt = '2021-06-21 12:00:00'
test_end_dt = '2021-06-21 17:00:00'

train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['vibration']]
test = energy.copy()[(energy.index >= test_start_dt) & (energy.index < test_end_dt)][['vibration']]

scaler = MinMaxScaler()

train['vibration'] = scaler.fit_transform(train)

test['vibration'] = scaler.transform(test)

train_data = train["vibration"].tolist()
test_data = test["vibration"].tolist()

train_dates = [int(x.timestamp()) for x in train.index]
test_dates = [int(x.timestamp()) for x in test.index]
train_dates = np.array(train_dates)
test_dates = np.array(test_dates)
train_dates = np.reshape(train_dates, (len(train_dates), 1))
test_dates = np.reshape(test_dates, (len(test_dates), 1))



# params = {
#     'kernel': ['rbf', 'linear', 'sigmoid'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
#     'epsilon': [0.001, 0.01, 0.1,  1, 10, 100]
#     }

params = {
    'kernel': ['linear'],
    'C': [0.001],
    'gamma': [0.001],
    'epsilon': [0.001]
    }

grid_search = GridSearchCV(SVR(), params, cv=5, n_jobs=-1,verbose=1)

grid_search.fit(train_dates, train_data)

print("train score - " + str(grid_search.score(train_dates, train_data)))
print("test score - " + str(grid_search.score(test_dates, test_data)))

print("SVR GridSearch score: "+str(grid_search.best_score_))
print("SVR GridSearch params: ")
print(grid_search.best_params_)