import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

time = np.arange(0, 1000, 0.5)
value = np.sin(time)

train, test = value[:800], value[-200:]
train_time, test_time = time[:800], time[-200:]

timesteps=10
train_time =train_time.reshape(-1,1)
test_time =test_time.reshape(-1,1)

train_data_timesteps = np.array([[j for j in train[i:i+timesteps]] for i in range(0, len(train)-timesteps+1)])[:, :,]

test_data_timesteps=np.array([[j for j in test[i:i+timesteps]] for i in range(0,len(test)-timesteps+1)])[:,:,]

x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

model = SVR(kernel='rbf',gamma=0.1, C=10, epsilon = 0.05)

model.fit(x_train, y_train[:,0])

y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)
print(train)
y_mov_test = [x for x in test[:9]]
for i in range(0, len(y_test_pred)):
    pred_i = model.predict([y_mov_test[-9:]])
    # pred_i = model.predict([[y_mov_test[-9], y_mov_test[-8], y_mov_test[-7], y_mov_test[-6], y_mov_test[-5],
    #                          y_mov_test[-4], y_mov_test[-3], y_mov_test[-2], y_mov_test[-1]]]).reshape(-1, 1)
    y_mov_test.append(pred_i[0])
y_mov_test = y_mov_test[-len(y_test_pred):]

print(len(y_mov_test))



fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(test_time,test, lw=3, label='Test Values')
ax.plot(test_time[-191:], y_test_pred,  lw=3, label='One Step Predictions')
ax.plot(test_time[-191:], y_mov_test, lw=3, label='Multi-Step Predictions')
ax.legend(loc="upper right")
plt.savefig("sine.eps", format="eps", dpi=1200)
plt.savefig("sine.jpg", format="jpg", dpi=1200)
plt.show()

