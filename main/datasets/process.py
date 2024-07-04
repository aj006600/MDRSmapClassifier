import os
import pandas as pd

def calculate_ema(data, n):
    return data.ewm(span=n, min_periods=n).mean()
def calculate_macd(data):
    ema12 = calculate_ema(data, 12)
    ema26 = calculate_ema(data, 26)
    return ema12 - ema26
def calculate_signal(macd):
    return calculate_ema(macd, 9)
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def mk_tech_f(ticker_data):
    ticker_data['EMA9'] = calculate_ema(ticker_data['close'], 9)
    ticker_data['EMA12'] = calculate_ema(ticker_data['close'], 12)
    ticker_data['EMA26'] = calculate_ema(ticker_data['close'], 26)
    ticker_data['MACD'] = calculate_macd(ticker_data['close'])
    ticker_data['Signal'] = calculate_signal(ticker_data['MACD'])
    ticker_data['RSI14'] = calculate_rsi(ticker_data['MACD'])
    return ticker_data

def diff_f(ticker_data):
    f_d = ['financing', 'fi', 'ii']
    ticker_data[f_d] = ticker_data[f_d].diff()
    ticker_data = ticker_data.dropna().reset_index(drop=True)
    return ticker_data

def processing(ticker_data):
    ticker_data = mk_tech_f(ticker_data)
    ticker_data = diff_f(ticker_data)
    return ticker_data