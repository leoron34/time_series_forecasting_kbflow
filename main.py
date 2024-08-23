from src.data_preprocessing import load_and_prepare_data
from src.hyperparameter_tuning import arima_grid_search
from src.model_training import train_arima_model, save_model

def main():
    # Step 1: Load and Prepare the Data
    filepath = 'data/stock_prices.csv'  # Update this with the actual path to your CSV file
    time_series = load_and_prepare_data(filepath)
    
    # Step 2: Optional - Hyperparameter Tuning
    print("Starting hyperparameter tuning...")
    best_model = arima_grid_search(time_series)
    print("Hyperparameter tuning completed.")

    # Step 3: Train the ARIMA Model
    print("Training ARIMA model...")
    arima_model = train_arima_model(time_series)
    print("Model training completed.")

    # Step 4: Save the Model
    model_save_path = 'models/arima_model.pkl'
    save_model(arima_model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()




# from src.data_preprocessing import preprocess_data
# from src.model_training import build_and_train_model
# from src.hyperparameter_tuning import hyperparameter_tuning

# # Preprocess the data
# X_train, y_train, X_test, y_test, scaler = preprocess_data('data/stock_prices.csv')

# # Hyperparameter tuning
# # best_hps = hyperparameter_tuning(X_train, y_train, X_test, y_test)

# # Train the model
# model = build_and_train_model(X_train, y_train, X_test, y_test)



# main.p
