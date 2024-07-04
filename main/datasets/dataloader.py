import os
import pandas as pd

# Get current file path #
def file_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '..', '..')

# Get ticker data #
def get_t_d(ticker):
    f_p = file_path()
    f_p = os.path.join(f_p, 'data', 'TaiwanStockData_perTicker', 'data', f'{ticker}.xlsx')
    t_d = pd.read_excel(f_p)
    return t_d

# Transform Traditional Chinese into English #
def f_tc_to_eng(t_d):
    # feature needed
    t_d = t_d[['交易日期', '開盤', '最高', '最低', '收盤', '成交量', '融資', '外資餘額', '投信餘額', '自營商買賣超', '主力買賣超', '資金流向']]
    # tc_to_eng
    t_d = t_d.rename(columns={'交易日期':'Date', '開盤':'open', '最高':'high', '最低':'low', '收盤':'close', '成交量':'volume',
                              '融資': 'financing', '外資餘額':'fi', '投信餘額':'ii', '自營商買賣超':'di', '主力買賣超':'rp',
                              '資金流向':'capital'})
    return t_d

# Obtain the ticker data of data format used by MDRSmapClassifier. #
def get_ticker_data(ticker, after_date):
    t_d = get_t_d(ticker)
    t_d = f_tc_to_eng(t_d)
    t_d['Date'] = pd.to_datetime(t_d['Date'])
    t_d = t_d[t_d['Date']>=after_date].reset_index(drop=True)
    ticker_data = t_d.copy()
    return ticker_data