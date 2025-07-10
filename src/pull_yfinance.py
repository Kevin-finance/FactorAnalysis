from settings import config
import yfinance as yf


DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")



def pull_closing_price(tickers:list,start_date = START_DATE, end_date = END_DATE, adjusted=True):
    close = yf.download(tickers,start_date,end_date,auto_adjust=adjusted)['Close']
    







# This srcipt is needed to pull out ohlcv for our universe



if __name__ == "__main__":
    print("hi")



