import numpy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from copy import deepcopy
from sklearn.svm import SVR


class PredictSVR:
    def __init__(self, train, test):
        self.timesteps = 5
        self.train = deepcopy(train)
        self.test = deepcopy(test)
        self.p_train, self.p_test, self.gp_test= self.predict_SVR()

    def predict_SVR(self):
        scaler = MinMaxScaler()
        self.train['value'] = scaler.fit_transform(self.train)
        self.test['value'] = scaler.transform(self.test)

        train_data = self.train.values
        test_data = self.test.values

        train_data_timesteps = np.array(
            [[j for j in train_data[i:i + self.timesteps]] for i in range(0, len(train_data) - self.timesteps + 1)])[:,
                               :, 0]

        test_data_timesteps = np.array(
            [[j for j in test_data[i:i + self.timesteps]] for i in range(0, len(test_data) - self.timesteps + 1)])[:, :,
                              0]

        x_train, y_train = train_data_timesteps[:, :self.timesteps - 1], train_data_timesteps[:, [self.timesteps - 1]]
        x_test, y_test = test_data_timesteps[:, :self.timesteps - 1], test_data_timesteps[:, [self.timesteps - 1]]

        # model = SVR(kernel='rbf', gamma=0.5, C=42, epsilon=0.05)
        model = SVR(kernel='rbf', gamma=0.1, C=10, epsilon=0.01)

        model.fit(x_train, y_train[:, 0])

        y_train_pred = model.predict(x_train).reshape(-1, 1)
        y_test_pred = model.predict(x_test).reshape(-1, 1)


        y_mov_test = [x[0] for x in train_data[-4:]]
        for i in range(0, len(self.test)):
            pred = model.predict([[y_mov_test[-4], y_mov_test[-3], y_mov_test[-2], y_mov_test[-1]]]).reshape(-1, 1)
            y_mov_test.append(pred[0][0])
        y_mov_test = y_mov_test[-len(self.test):]
        y_mov_test = np.array(y_mov_test).reshape(-1, 1)

        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        y_mov_test = scaler.inverse_transform(y_mov_test)

        return y_train_pred, y_test_pred, y_mov_test
