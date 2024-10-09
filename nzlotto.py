import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

def load_historical_data(file_path):
    # Load data from a CSV file
    # Assuming your CSV has columns: DrawDate, Ball 1, Ball 2, Ball 3, Ball 4, Ball 5, Ball 6, Bonus Ball
    data = pd.read_csv(file_path, parse_dates=['DrawDate'])
    # Sort by date to ensure chronological order
    data = data.sort_values('DrawDate')
    # Select only the number columns
    number_columns = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Ball 6', 'Bonus Ball']
    return data[number_columns]

def prepare_data(data, look_back=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back)])
        y.append(scaled_data[i + look_back])
    return np.array(X), np.array(y), scaler

def build_model(look_back, features):
    model = Sequential([
        Input(shape=(look_back, features)),
        LSTM(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(100, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(100, activation='relu'),
        Dropout(0.2),
        Dense(features)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def generate_lotto_numbers(model, last_numbers, scaler):
    input_data = np.array([last_numbers])
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 7))
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction)[0]
    main_numbers = sorted([round(num) for num in prediction[:6]])
    bonus_number = round(prediction[6])
    return main_numbers, bonus_number

def main():
    # Load historical data
    data = load_historical_data('path_to_your_csv_file.csv')
    
    look_back = 10  # You can adjust this based on how many past draws you want to consider
    X, y, scaler = prepare_data(data, look_back)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(look_back, 7)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr], verbose=1)
    
    # Use the last 'look_back' number of draws to predict the next one
    last_numbers = scaler.transform(data.iloc[-look_back:].values)
    main_numbers, bonus_number = generate_lotto_numbers(model, last_numbers, scaler)
    
    print("Predicted Lotto Numbers:")
    print("Main Numbers:", " ".join(map(str, main_numbers)))
    print("Bonus Number:", bonus_number)

if __name__ == "__main__":
    main()