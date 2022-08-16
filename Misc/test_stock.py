import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR

msft = yf.Ticker("MSFT")

# get stock info

def get_data(df):
    data = df.copy()
    data['date'] = data['date'].astype(str).str.split('-').str[2]
    data['date'] = pd.to_numeric(data['date'])
    return [ data['date'].tolist(), data['Close'].tolist() ] # Convert


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    x = np.reshape(x, (len(x), 1))
    # print(dates,prices)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Fit regression model

    svr_rbf.fit(dates, prices)
    print(x)
    plt.scatter(dates, prices, c='k', label='Data')

    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')


    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0]

# get historical market data
#hist = msft.history(period="30d")
hist = msft.history(start="2022-06-02", end="2022-07-01")
hist['date'] = hist.index




dates, prices = get_data(hist)
print(dates, prices)

predicted_price = predict_prices(dates, prices, [31])
# print(predicted_price)
# dist = msft.history(start="2022-07-01", end="2022-07-01")
# print(dist["Close"])
