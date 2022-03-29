from scipy.stats import norm  # For generating random numbers
import matplotlib.pyplot as plt  # For ploting purpose
import numpy as np
import yfinance as yf


def geoBrownianMotion(sigma, mu, x, N, T):

    # Step 0: Initial Condition
    dt = float(T) / N
    S = x
    S_all = [S]

    # Step 1-4:
    for i in range(1, N):
        dWt = np.random.normal(loc=0, scale=np.sqrt(dt))
        S = S * np.exp((mu - float(sigma**2) / 2) * dt + sigma * (dWt))
        S_all.append(S)
    return S_all

def brownianMotion(sigma, mu, N, T):
    dt = float(T) / N
    dWt = np.random.normal(loc=0, scale=np.sqrt(dt))
    S_all = []

    for i in range(1, N):
        dWt = np.random.normal(loc=0, scale=np.sqrt(dt))
        S = mu * dt + sigma * dWt
        S_all.append(S)
    return S_all

def fitModel(data):
   return 

def predictPrice(stock):
    info = yf.Ticker(stock)
    data = info.history(period="10y")
    prices = data['Close']
    fitModel(prices)


# plt.plot(brownianMotion(.5, 2, 1000, 5))
# plt.show()
