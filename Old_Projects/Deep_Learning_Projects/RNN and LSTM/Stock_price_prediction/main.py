# %%
import pandas as pd
import streamlit as st
from stock_price_modules.helper_module import get_close_prediction, get_OHLC_today

# %%
st.title("Stock Forecast WebApp")
accuracy = [('BTC-USD', 	7784.548754479958),
('ETH-USD', 	136.31814167823322),
('USDT-USD', 	0.004122312790081825), 
('BNB-USD', 	7.789193747956076),
('USDC-USD', 	0.003766464815993253)]
accuracy = pd.DataFrame(accuracy, columns=["Ticker", "Mean Absolute Error($)"])
with st.sidebar:
    st.subheader("Model Accuracy")
    st.table(accuracy)

    st.subheader("Disclaimer")
    st.write("The forecast given on this app is not financial advice and is for educational or research purposes")

# %%
ticker = st.radio(
     "Please select a ticker to use",
     (None, 'BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD'))

# %%
if st.button('Get OHLC values for today'):
    values = get_OHLC_today(ticker, for_prediction=False)
    #check to see that it doesn't return empty data
    if values:
        #check if ticker has been selected
        st.error("Please select a ticker") if ticker is None else st.table(values)
    else:
        st.error("empty data receieved, market may not be open today")

# %%
if st.button('Get closing price forecast for today'):
    if ticker is None:
        values = get_OHLC_today(ticker, for_prediction=True)
        #check to see that it doesn't return empty data
        if values:
            pred = get_close_prediction(ticker, for_prediction=True)
            st.write(f"The forecast for today is **{pred}** dollars")
        else:
            st.error("empty data receieved, market may not be open today")
            st.info("try imputing the values if you have them")
    else:
        st.error("Please select a ticker")

# %%
def check_impute(ohlv): #checks that manual values were imputed
    if ohlv[0] !=0 and ohlv[1] != 0 and ohlv[2] != 0 and ohlv[3] != 0:
        return True
    else:
        return False

# %%
st.subheader("If you would like to impute your own values, do so below")
open = st.number_input('Insert Open')
high = st.number_input('Insert High')
low = st.number_input('Insert Low')
volume = st.number_input('Insert Volume')
ohlv = [open, high, low, volume]

#check that ticker is selected and values were imputed
if ticker is not None and check_impute(ohlv):
    pred2 = get_close_prediction(ticker, ohlv)
    st.markdown(f"The forecast for today is **{pred2}** dollars")
else:
    st.error("Please make sure ticker is selected and all values were imputed")



