import pandas as pd
from datetime import datetime

# Labeling #
def labeling(ticker_data, TP):
    # SMA-P/P, binary classification
    ticker_data[f'y_{TP}'] = ticker_data['close'].rolling(window=TP).mean()
    ticker_data[f'y_{TP}'] = ticker_data[f'y_{TP}'].shift(-TP)
    ticker_data = ticker_data.dropna().reindex()
    ticker_data[f'y_{TP}'] = ((ticker_data[f'y_{TP}'] - ticker_data['close']) >= 0).astype(int)

    origi_data = ticker_data.copy() # dataframe for validation
    # restore to a dataframe without disclosing future information
    ticker_data[f'y_{TP}'] = ticker_data[f'y_{TP}'].shift(TP)
    ticker_data = ticker_data.dropna().reset_index(drop=True)

    return origi_data, ticker_data

# Get every single dates in data #
def unique_date(ticker_data):
    if not pd.api.types.is_datetime64_any_dtype(ticker_data['Date']):
        ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
    unique_dates = ticker_data['Date'].dt.strftime('%Y-%m-%d').unique()
    unique_dates = list(unique_dates)
    return unique_dates

# Splite Library and Prediction #
def splite_l_p(ticker_data, unique_dates, start_pred_date):
    if start_pred_date in unique_dates:
        actual_start_pred_date = start_pred_date
        before_start_pred_date = unique_dates[unique_dates.index(start_pred_date)-1]
    else:
        start_pred_date = datetime.strptime(start_pred_date, '%Y-%m-%d')
        filtered_dates = [date for date in unique_dates if datetime.strptime(date, '%Y-%m-%d') > start_pred_date]
        actual_start_pred_date = min(filtered_dates, key=lambda date: datetime.strptime(date, '%Y-%m-%d'))
        before_start_pred_date = unique_dates[unique_dates.index(actual_start_pred_date)-1]
    Library = ticker_data[ticker_data['Date'] <= before_start_pred_date]
    Prediction = ticker_data[ticker_data['Date'] >= unique_dates[unique_dates.index(actual_start_pred_date)]]
    return Library, Prediction, actual_start_pred_date

# Data preprocessing #
def data_preprocessing(ticker_data, tp, start_pred_date):
    origi_data, ticker_data = labeling(ticker_data, tp)
    unique_dates = unique_date(ticker_data)
    Library, Prediction, actual_start_pred_date = splite_l_p(ticker_data, unique_dates, start_pred_date)
    return origi_data, Library, Prediction, actual_start_pred_date