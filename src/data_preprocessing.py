# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

# def preprocess_data(file_path):
#     df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

#     # Feature Engineering
#     df['Day'] = df.index.day
#     df['Month'] = df.index.month
#     df['Year'] = df.index.year
#     df['DayOfWeek'] = df.index.dayofweek
#     df['Lag_1'] = df['Close'].shift(1)
#     df['Lag_7'] = df['Close'].shift(7)
#     df.dropna(inplace=True)

#     # Train-Test Split
#     train, test = train_test_split(df, test_size=0.2, shuffle=False)

#     # Scaling features
#     scaler = StandardScaler()
#     train_scaled = scaler.fit_transform(train)
#     test_scaled = scaler.transform(test)

#     X_train = train_scaled[:, :-1]
#     y_train = train_scaled[:, -1]
#     X_test = test_scaled[:, :-1]
#     y_test = test_scaled[:, -1]

#     return X_train, y_train, X_test, y_test, scaler


# data_preprocessing.py
import pandas as pd

def load_and_prepare_data(filepath):
    # Load the stock data
    data = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    
    # Ensure the data is sorted by date
    data = data.sort_index()

    # Set the frequency to business days (B) or another appropriate frequency
    data = data.asfreq('B')

    # Use the 'Close' price for ARIMA modeling
    time_series = data['Close'].dropna()  # Dropping NA values if any

    return time_series


# Example usage:
# time_series = load_and_prepare_data('data/stock_prices.csv')
