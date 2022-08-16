

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from ProcessDB import ProcessDB
from PredictSVR import PredictSVR
from statsmodels.tsa.statespace.sarimax import SARIMAX




database_path = "/home/sachin/Downloads/RWO_0004_Ventilatoren_00.sqlite"

forecast_hour = '2021-05-27 12:00:00'
training_duration = 6

scaler = MinMaxScaler()

df = ProcessDB(database_path, forecast_hour, training_duration)

#%%

scaler = MinMaxScaler()

df.train['value'] = scaler.fit_transform(df.train)
df.test['value'] = scaler.transform(df.test)

HORIZON = 904

order = (4, 1, 0)
seasonal_order = (1, 1, 0, 24)

model = SARIMAX(endog=df.train, order=order, seasonal_order=seasonal_order)
results = model.fit()

print(results.summary())

#%%

yhat = results.forecast(steps = HORIZON)


#%%

yhat = scaler.inverse_transform(yhat.values.reshape(-1,1))
print(yhat)
