import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from keras_tuner import RandomSearch
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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


def prepare_data(data, look_back=24):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back)])
        y.append(scaled_data[i + look_back])
    return np.array(X), np.array(y), scaler


def build_model(hp):
    look_back = hp.Int('look_back', 16, 52, step=2)
    model = Sequential()
    model.add(LSTM(hp.Int('lstm_units_1', 32, 512, step=32), 
                   activation='relu', 
                   return_sequences=True, 
                   input_shape=(look_back, 7)))
    model.add(Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)))
    
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        model.add(LSTM(hp.Int(f'lstm_units_{i+2}', 32, 512, step=32), 
                       activation='relu', 
                       return_sequences=True))
        model.add(Dropout(hp.Float(f'dropout_{i+2}', 0, 0.5, step=0.1)))
    
    model.add(LSTM(hp.Int('lstm_units_last', 32, 512, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_last', 0, 0.5, step=0.1)))
    model.add(Dense(7, activation='sigmoid'))
    
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG')),
                  loss='mse')
    return model


def generate_lotto_numbers(model, last_numbers, scaler):
    input_data = np.array([last_numbers])
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction)[0]
    
    # Ensure numbers are within valid range (e.g., 1-40)
    valid_range = lambda x: max(1, min(round(x), 40))
    main_numbers = sorted(set([valid_range(num) for num in prediction[:6]]))
    
    # If we don't have enough unique numbers, add some
    while len(main_numbers) < 6:
        main_numbers.add(random.randint(1, 40))
    
    main_numbers = sorted(main_numbers)
    bonus_number = valid_range(prediction[6])
    
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


def evaluate_model(model, X_test, y_test, scaler, data, look_back, actual_numbers, actual_bonus):
    predictions = model.predict(X_test)
    
    # Check for NaNs in predictions
    if np.isnan(predictions).any():
        print("Warning: NaN values found in predictions")
        print("Number of NaN values:", np.isnan(predictions).sum())
        print("Positions of NaN values:", np.where(np.isnan(predictions)))
        return float('inf'), 0, 0  # Return worst possible score
    
    mse = mean_squared_error(y_test, predictions)
    
    # Generate a prediction for the actual winning numbers
    last_numbers = scaler.transform(data.iloc[-look_back:].values)
    main_numbers, bonus_number = generate_lotto_numbers(model, last_numbers, scaler)
    
    # Calculate how many numbers match the actual winning numbers
    matches = len(set(main_numbers) & set(actual_numbers))
    bonus_match = 1 if bonus_number == actual_bonus else 0
    
    return mse, matches, bonus_match


def cross_validate_model(data, hp, actual_numbers, actual_bonus, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    match_scores = []
    
    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        X_train, y_train, scaler = prepare_data(train_data, look_back=24)
        X_test, y_test, _ = prepare_data(test_data, look_back=24)
        
        model = build_model(hp)
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        mse, matches, _ = evaluate_model(model, X_test, y_test, scaler, test_data, hp.get('look_back'), actual_numbers, actual_bonus)
        mse_scores.append(mse)
        match_scores.append(matches)
    
    return np.mean(mse_scores), np.mean(match_scores)


def run_trials(data, actual_numbers, actual_bonus):
    tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=3,
    directory='my_dir',
    project_name='lottery_prediction'
    )

    X, y, scaler = prepare_data(data, look_back=24)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    results = []
    for hp in best_hps:
        mse, matches = cross_validate_model(data, hp, actual_numbers, actual_bonus, n_splits=5)
        results.append({
            'look_back': hp.get('look_back'),
            'lstm_units': hp.get('lstm_units'),
            'dropout': hp.get('dropout'),
            'learning_rate': hp.get('learning_rate'),
            'l1': hp.get('l1'),
            'l2': hp.get('l2'),
            'mse': mse,
            'matches': matches
        })
    
    return results


def main():
    # Load and prepare data
    data = load_historical_data('data.csv')
    max_number = 40
    data = data.applymap(lambda x: max(1, min(x, max_number)))
    if data is None:
        print("Failed to load data. Exiting.")
        return

    analyze_data(data)

    # Prepare data for model
    X, y, scaler = prepare_data(data, look_back=24)  # You might want to make look_back a tunable parameter
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the hyperparameter search
    tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=100,  # Increased from 50
    executions_per_trial=3,
    directory='my_dir',
    project_name='lottery_prediction'
    )

    # Use cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
                    callbacks=[EarlyStopping(patience=10)])

    # Perform the hyperparameter search
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)

    # Train the final model
    history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), verbose=1)

    # Plot the training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Generate predictions
    last_numbers = scaler.transform(data.iloc[-best_hps.get('look_back'):].values)
    main_numbers, bonus_number = generate_lotto_numbers(model, last_numbers, scaler)

    print("\nPredicted Lotto Numbers:")
    print("Main Numbers:", " ".join(map(str, main_numbers)))
    print("Bonus Number:", bonus_number)

    print("\nMultiple Predictions:")
    for i in range(5):
        main_numbers, bonus_number = generate_lotto_numbers(model, last_numbers, scaler)
        print(f"Prediction {i+1}: Main Numbers: {' '.join(map(str, main_numbers))}, Bonus Number: {bonus_number}")
        new_numbers = pd.DataFrame(np.array(main_numbers + [bonus_number]).reshape(1, -1), columns=data.columns)
        last_numbers = np.vstack((last_numbers[1:], scaler.transform(new_numbers)))

    print("\nPredictions completed.")

if __name__ == "__main__":
    main()
