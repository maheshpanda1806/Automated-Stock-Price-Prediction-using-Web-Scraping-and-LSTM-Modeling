import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import schedule
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    data = df['Close'].values.reshape(-1, 1)  # Use the 'Close' price for prediction
    return data

# Create dataset for LSTM
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Build LSTM model
def build_lstm_model(time_step):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and save the model for Apple (AAPL)
def train_and_save_model_apple():
    file_path = "AAPL_stock_data.csv"
    model_filename = "AAPL_lstm_model.pkl"
    
    # Load data and preprocess
    data = load_data(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Build, train, and save the model
    model = build_lstm_model(time_step)
    model.fit(X, Y, epochs=10, batch_size=32, verbose=1)
    
    # Save the model
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Apple (AAPL) model saved to {model_filename}")

# Train and save the model for Goldman Sachs (GS)
def train_and_save_model_goldman():
    file_path = "GS_stock_data.csv"
    model_filename = "GS_lstm_model.pkl"
    
    data = load_data(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = build_lstm_model(time_step)
    model.fit(X, Y, epochs=10, batch_size=32, verbose=1)
    
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Goldman Sachs (GS) model saved to {model_filename}")

# Train and save the model for Morgan Stanley (MS)
def train_and_save_model_morgan():
    file_path = "MS_stock_data.csv"
    model_filename = "MS_lstm_model.pkl"
    
    data = load_data(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = build_lstm_model(time_step)
    model.fit(X, Y, epochs=10, batch_size=32, verbose=1)
    
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Morgan Stanley (MS) model saved to {model_filename}")

# Train and save the model for Deutsche Bank (DB)
def train_and_save_model_db():
    file_path = "DB_stock_data.csv"
    model_filename = "DB_lstm_model.pkl"
    
    data = load_data(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = build_lstm_model(time_step)
    model.fit(X, Y, epochs=10, batch_size=32, verbose=1)
    
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Deutsche Bank (DB) model saved to {model_filename}")

# Scheduling the tasks to run daily at specific times
schedule.every().day.at("10:00").do(train_and_save_model_db)
schedule.every().day.at("10:05").do(train_and_save_model_morgan)
schedule.every().day.at("10:10").do(train_and_save_model_apple)
schedule.every().day.at("10:15").do(train_and_save_model_goldman)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check for scheduled tasks every minute