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


def prepare_data(data, look_back=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back)])
        y.append(scaled_data[i + look_back])
    return np.array(X), np.array(y), scaler


def build_model_tuner(hp):
    model = Sequential([
        Input(shape=(hp.Int('look_back', 5, 20), 7)),
        LSTM(hp.Int('lstm_units', 100, 150), activation='relu', return_sequences=True, 
             kernel_regularizer=l1_l2(l1=hp.Float('l1', 1e-5, 1e-2), l2=hp.Float('l2', 1e-5, 1e-2))),
        Dropout(hp.Float('dropout', 0.2, 0.3)),
        LSTM(hp.Int('lstm_units', 100, 150), activation='relu', return_sequences=True,
             kernel_regularizer=l1_l2(l1=hp.Float('l1', 1e-5, 1e-2), l2=hp.Float('l2', 1e-5, 1e-2))),
        Dropout(hp.Float('dropout', 0.2, 0.3)),
        LSTM(hp.Int('lstm_units', 100, 150), activation='relu',
             kernel_regularizer=l1_l2(l1=hp.Float('l1', 1e-5, 1e-2), l2=hp.Float('l2', 1e-5, 1e-2))),
        Dropout(hp.Float('dropout', 0.2, 0.3)),
        Dense(7)
    ])
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-1, sampling='log')), loss='mse')
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


def cross_validate_model(data, hp, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    match_scores = []
    
    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        X_train, y_train, scaler = prepare_data(train_data, hp.get('look_back'))
        X_test, y_test, _ = prepare_data(test_data, hp.get('look_back'))
        
        model = build_model_tuner(hp)
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        mse, matches, _ = evaluate_model(model, X_test, y_test, scaler, test_data, hp.get('look_back'), actual_numbers, actual_bonus)
        mse_scores.append(mse)
        match_scores.append(matches)
    
    return np.mean(mse_scores), np.mean(match_scores)


def run_trials(data, actual_numbers, actual_bonus):
    tuner = RandomSearch(
        build_model_tuner,
        objective='val_loss',
        max_trials=50,
        executions_per_trial=3,
        directory='my_dir',
        project_name='lottery_prediction'
    )

    X, y, _ = prepare_data(data, look_back=10)  # Default look_back, will be tuned
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[ReduceLROnPlateau()])

    best_hps = tuner.get_best_hyperparameters(num_trials=5)
    
    results = []
    for hp in best_hps:
        mse, matches = cross_validate_model(data, hp, n_splits=5)
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
    data = load_historical_data('data.csv')
    if data is None:
        print("Failed to load data. Exiting.")
        return

    analyze_data(data)

    actual_numbers = [1, 3, 10, 24, 32, 38]
    actual_bonus = 3
    results = run_trials(data, actual_numbers, actual_bonus)

    sorted_results = sorted(results, key=lambda x: (-x['matches'], x['mse']))

    print("\nTop 5 Parameter Combinations:")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. look_back={result['look_back']}, lstm_units={result['lstm_units']}, "
              f"dropout={result['dropout']}, learning_rate={result['learning_rate']}")
        print(f"   L1={result['l1']:.6f}, L2={result['l2']:.6f}")
        print(f"   Avg Matches: {result['matches']:.2f}, MSE: {result['mse']:.6f}\n")

    best_hp = sorted_results[0]
    X, y, scaler = prepare_data(data, best_hp['look_back'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = build_model_tuner(best_hp)
    history = best_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[ReduceLROnPlateau()], verbose=1)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    last_numbers = scaler.transform(data.iloc[-best_hp['look_back']:].values)
    main_numbers, bonus_number = generate_lotto_numbers(best_model, last_numbers, scaler)

    print("\nPredicted Lotto Numbers:")
    print("Main Numbers:", " ".join(map(str, main_numbers)))
    print("Bonus Number:", bonus_number)

    print("\nMultiple Predictions:")
    for i in range(5):
        main_numbers, bonus_number = generate_lotto_numbers(best_model, last_numbers, scaler)
        print(f"Prediction {i+1}: Main Numbers: {' '.join(map(str, main_numbers))}, Bonus Number: {bonus_number}")
        new_numbers = np.array(main_numbers + [bonus_number]).reshape(1, -1)
        last_numbers = np.vstack((last_numbers[1:], scaler.transform(new_numbers)))      


if __name__ == "__main__":
    main()
