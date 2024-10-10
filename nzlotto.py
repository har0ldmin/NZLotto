import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

def load_historical_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['DrawDate'])
        print("Columns in the CSV file:", data.columns.tolist())
        
        column_mapping = {
            'Ball1': 'Ball 1',
            'Ball2': 'Ball 2',
            'Ball3': 'Ball 3',
            'Ball4': 'Ball 4',
            'Ball5': 'Ball 5',
            'Ball6': 'Ball 6',
            'Bonus Ball': 'Bonus Ball'
        }
        
        data = data.rename(columns=column_mapping)
        
        number_columns = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Ball 6', 'Bonus Ball']
        return data[number_columns]
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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

def analyze_data(data):
    # Frequency analysis
    all_numbers = data.iloc[:, :6].values.flatten()
    
    # Calculate frequency of each number and sort in descending order
    number_frequency = pd.Series(all_numbers).value_counts().sort_values(ascending=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(24, 10))  # Increased width to accommodate text
    sns.histplot(all_numbers, bins=range(1, max(all_numbers)+2), kde=False, discrete=True, ax=ax)
    ax.set_title('Distribution of Winning Numbers', fontsize=16)
    ax.set_xlabel('Number', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xticks(range(1, max(all_numbers)+1))
    ax.set_yticks(range(0, max(number_frequency)+1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add summary title
    ax.text(1.02, 1.0, "Winning Balls Frequencies:\n\n\n", 
            transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # Add text summary on the right side, sorted by frequency with color coding
    y_position = 0.95
    for i, (num, freq) in enumerate(number_frequency.items()):
        if i < 7:
            color = 'red'
        elif i >= len(number_frequency) - 7:
            color = 'blue'
        else:
            color = 'black'
        ax.text(1.02, y_position, f"Ball {num}: {freq}", transform=ax.transAxes, 
                fontsize=10, color=color, fontweight='bold')
        y_position -= 0.03
    
    plt.tight_layout()
    plt.show()

    # Correlation analysis
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Ball Positions')
    plt.show()

    # Time series analysis
    data_with_index = data.copy()
    data_with_index['Draw Number'] = range(len(data))
    plt.figure(figsize=(12, 8))
    for col in data.columns:
        plt.plot(data_with_index['Draw Number'], data_with_index[col], label=col)
    plt.title('Number Trends Over Time')
    plt.xlabel('Draw Number')
    plt.ylabel('Ball Value')
    plt.legend()
    plt.show()

def main():
    # Load historical data
    data = load_historical_data('data.csv')
    
    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Print basic information about the data
    print("\nFirst few rows of the loaded data:")
    print(data.head())
    
    print("\nShape of the data:")
    print(data.shape)
    
    print("\nColumn names:")
    print(data.columns)

    # Perform data analysis
    analyze_data(data)

    # Prepare data for LSTM
    look_back = 10  # You can adjust this based on how many past draws you want to consider
    X, y, scaler = prepare_data(data, look_back)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model(look_back, 7)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr], verbose=1)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # # Generate predictions
    # last_numbers = scaler.transform(data.iloc[-look_back:].values)
    # main_numbers, bonus_number = generate_lotto_numbers(model, last_numbers, scaler)
    
    # print("Predicted Lotto Numbers:")
    # print("Main Numbers:", " ".join(map(str, main_numbers)))
    # print("Bonus Number:", bonus_number)

    # # Generate multiple predictions
    # num_predictions = 5
    # print("\nMultiple Predictions:")
    # for i in range(num_predictions):
    #     main_numbers, bonus_number = generate_lotto_numbers(model, last_numbers, scaler)
    #     print(f"Prediction {i+1}: Main Numbers: {' '.join(map(str, main_numbers))}, Bonus Number: {bonus_number}")
    #     # Update last_numbers for next prediction
    #     new_numbers = np.array(main_numbers + [bonus_number]).reshape(1, -1)
    #     last_numbers = np.vstack((last_numbers[1:], scaler.transform(new_numbers)))

if __name__ == "__main__":
    main()
