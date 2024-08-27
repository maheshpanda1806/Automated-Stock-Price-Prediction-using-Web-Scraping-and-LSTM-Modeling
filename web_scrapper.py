import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import schedule

def save_stock_data_Apple():
    # Specify the stock ticker (e.g., Apple Inc.)
    ticker = "AAPL"
    
    # Fetch historical data for the last 5 years
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y", interval="1d")  # Adjust the period and interval if needed

    # Save the data to a CSV file with the current date as the filename
    apple_stocks = f"{ticker}_stock_data_{datetime.today().strftime('%Y-%m-%d')}.csv"
    hist.to_csv(apple_stocks)
    print(f"Data saved to {apple_stocks}")

def save_stock_data_Goldman():
    ticker = "GS"
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y", interval="1d")  

    stocks = f"{ticker}_stock_data_{datetime.today().strftime('%Y-%m-%d')}.csv"
    hist.to_csv(stocks)
    print(f"Data saved to {stocks}")

def save_stock_data_Morgan():
    ticker = "MS"
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y", interval="1d") 

    apple_stocks = f"{ticker}_stock_data_{datetime.today().strftime('%Y-%m-%d')}.csv"
    hist.to_csv(apple_stocks)
    print(f"Data saved to {apple_stocks}")

def save_stock_data_db():
    # Specify the stock ticket
    ticker = "DB"
    
    # Fetch historical data for the last 5 years
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y", interval="1d")  

    apple_stocks = f"{ticker}_stock_data_{datetime.today().strftime('%Y-%m-%d')}.csv"
    hist.to_csv(apple_stocks)
    print(f"Data saved to {apple_stocks}")

# Scheduling the tasks to run daily at specific times
schedule.every().day.at("09:00").do(save_stock_data_Apple)
schedule.every().day.at("09:05").do(save_stock_data_db)
schedule.every().day.at("09:10").do(save_stock_data_Goldman)
schedule.every().day.at("09:15").do(save_stock_data_Morgan)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check for scheduled tasks every minute