import pmdarima as pm
import joblib

def train_arima_model(time_series):
    # Automatically select the best ARIMA model using pmdarima
    model = pm.auto_arima(time_series, seasonal=False, stepwise=True, suppress_warnings=True)


    # Fit the model
    model.fit(time_series)
    
    return model

def save_model(model, filepath='models/arima_model.pkl'):
    # Save the ARIMA model
    joblib.dump(model, filepath)



# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout


# def build_and_train_model(X_train, y_train, X_test, y_test):
#     X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
#     X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=25))
#     model.add(Dense(units=1))

#     model.compile(optimizer='adam', loss='mean_squared_error')

#     model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
#     model.save('models/lstm_model.h5')

#     return model


# model_training.py

