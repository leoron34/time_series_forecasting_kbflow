import yfinance as yf

# Download data for a specific stock
ticker = 'AAPL'  # Replace with your desired stock ticker symbol
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Save the data to a CSV file
data.to_csv('data/stock_prices.csv')

print("Data downloaded and saved to 'data/stock_prices.csv'")
