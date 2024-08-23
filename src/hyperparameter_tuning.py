import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def arima_grid_search(time_series):
    p = range(0, 4)
    d = range(0, 2)
    q = range(0, 4)
    pdq = [(x, y, z) for x in p for y in d for z in q]
    
    best_aic = np.inf
    best_params = None
    best_model = None
    
    for param in pdq:
        try:
            model = ARIMA(time_series, order=param)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = param
                best_model = results
        except:
            continue
    
    print(f'Best ARIMA model order: {best_params} with AIC: {best_aic}')
    return best_model


# import kerastuner as kt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout


# def hyperparameter_tuning(X_train, y_train, X_test, y_test):
#     def build_model(hp):
#         model = Sequential()
#         model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), 
#                        return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#         model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
#         model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False))
#         model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
#         model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=256, step=16)))
#         model.add(Dense(1))

#         model.compile(optimizer='adam', loss='mean_squared_error')
#         return model

#     tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=5, executions_per_trial=3, directory='my_dir', project_name='lstm_tuning')

#     tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

#     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

#     return best_hps
