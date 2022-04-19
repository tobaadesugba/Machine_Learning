# %%
import yfinance as yf
from datetime import date
from keras.models import load_model

# %%
def get_OHLC_today(ticker, for_prediction=False):
    #get OHLC for ticker for today
    today = date.today().strftime("%Y-%m-%d")
    ticker_data = yf.download(
        tickers=ticker,
        start=today, end=today,
        progress=False)

    ticker_data['Date'] = ticker_data.index
    ticker_data = ticker_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    ticker_data.reset_index(inplace=True, drop=True)   
    
    #check to see if dataframe is empty
    if ticker_data.empty:
        return False ##-1 is used as placeholder for empty data
        
    #remove close and adj close value for prediction
    if for_prediction:
        ticker_data = ticker_data.drop(['Close', 'Adj Close'], axis=1)
        return list(ticker_data.loc[0])

    #return as list in format [open, high, low, close, adj close, volume]
    return ticker_data
    
# %%
def get_close_prediction(ticker, OHLV_list):
    #load model
    model = load_model(ticker+"_no_adjustments")
    #make predictions
    pred = model.predict(OHLV_list)
    #return predictions
    return pred