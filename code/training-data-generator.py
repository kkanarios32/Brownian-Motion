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

def brownianMotion(sigma, mu, x, N, T):
    dt = float(T) / N
    dWt = np.random.normal(loc=0, scale=np.sqrt(dt))
    S_all = []

    for i in range(1, N):
        dWt = np.random.normal(loc=0, scale=np.sqrt(dt))
        S = x + mu * dt + sigma * dWt
        S_all.append(S)
    return S_all

def fitModel(data, model):
    if model:
        data = np.log(data)
    time = np.linspace(0,len(data), num=len(data))
    A = np.vstack([time, np.ones(len(time))]).T
    alpha = np.linalg.lstsq(A, data, rcond=None)
    mu = alpha[0][0]
    diff = []
    for i in range(0, len(time)):
        diff.append(data[i] - (mu * i))
    diff = np.array(diff)
    sigma = np.std(diff, ddof = 1)
    if model:
        model = geoBrownianMotion(sigma, mu, data[0], len(data), len(data))
    else:
        model = brownianMotion(sigma, mu, data[0], len(data), len(data))
    return model



def predictPrice(stock, model):
    info = yf.Ticker(stock)
    data = info.history(period="1y")
    prices = np.array(data['Close'])
    get_error(prices)

def get_error(prices):
    N = 1000
    error = 0
    for i in range(0,N):
        fit = fitModel(prices, 1)
        curr_err = np.mean(np.subtract(prices,fit))
        plt.plot(fit)
        error = error + (curr_err/N)
    print(error)
# predictPrice("fb",1)
get_error(geoBrownianMotion(.5,.05,10000,252,1))
plt.show()
