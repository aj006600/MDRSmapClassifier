import os
import pandas as pd

# Technical indicator #
def calculate_ema(data, n): # EMA
    return data.ewm(span=n, min_periods=n).mean()
def calculate_macd(data): # MACD
    ema12 = calculate_ema(data, 12)
    ema26 = calculate_ema(data, 26)
    return ema12 - ema26
def calculate_signal(macd): # Signal
    return calculate_ema(macd, 9)
def calculate_rsi(data, period=14): # RSI
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Make technical indicator features #
def mk_tech_f(ticker_data):
    ticker_data['EMA9'] = calculate_ema(ticker_data['close'], 9)
    ticker_data['EMA12'] = calculate_ema(ticker_data['close'], 12)
    ticker_data['EMA26'] = calculate_ema(ticker_data['close'], 26)
    ticker_data['MACD'] = calculate_macd(ticker_data['close'])
    ticker_data['Signal'] = calculate_signal(ticker_data['MACD'])
    ticker_data['RSI14'] = calculate_rsi(ticker_data['MACD'])
    return ticker_data

# Differ processing #
def diff_f(ticker_data):
    f_d = ['financing', 'fi', 'ii']
    ticker_data[f_d] = ticker_data[f_d].diff()
    ticker_data = ticker_data.dropna().reset_index(drop=True)
    return ticker_data

# Processing #
def processing(ticker_data):
    ticker_data = mk_tech_f(ticker_data)
    ticker_data = diff_f(ticker_data)
    return ticker_data











def file_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '..', '..')

def get_t_d(ticker):
    f_p = file_path()
    f_p = os.path.join(f_p, 'data', 'raw', 'TaiwanStockData_perTicker', 'data', f'{ticker}.xlsx')
    t_d = pd.read_excel(f_p)
    return t_d

def f_tc_to_eng(t_d):
    # feature needed
    t_d = t_d[['交易日期', '開盤', '最高', '最低', '收盤', '成交量', '融資', '外資餘額', '投信餘額', '自營商買賣超', '主力買賣超', '資金流向']]
    # tc_to_eng
    t_d = t_d.rename(columns={'交易日期':'Date', '開盤':'open', '最高':'high', '最低':'low', '收盤':'close', '成交量':'volume',
                              '融資': 'financing', '外資餘額':'fi', '投信餘額':'ii', '自營商買賣超':'di', '主力買賣超':'rp',
                              '資金流向':'capital'})
    return t_d

def get_ticker_data(ticker, after_date):
    t_d = get_t_d(ticker)
    t_d = f_tc_to_eng(t_d)
    t_d['Date'] = pd.to_datetime(t_d['Date'])
    t_d = t_d[t_d['Date']>=after_date].reset_index(drop=True)
    ticker_data = t_d.copy()
    return ticker_data