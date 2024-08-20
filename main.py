from src.data_preprocessing import preprocess_data
from src.model_training import build_and_train_model
from src.hyperparameter_tuning import hyperparameter_tuning

# Preprocess the data
X_train, y_train, X_test, y_test, scaler = preprocess_data('data/stock_prices.csv')

# Hyperparameter tuning
best_hps = hyperparameter_tuning(X_train, y_train, X_test, y_test)

# Train the model
model = build_and_train_model(X_train, y_train, X_test, y_test)