import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def calculate_metrics(actual, predicted):
    """Calculate various performance metrics"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Basic metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Directional accuracy
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 2),
        'Directional Accuracy': round(directional_accuracy, 2),
        'R²': round(r_squared, 4)
    }
    
    return metrics

def backtest_model(data, model, train_size=0.8, forecast_days=7):
    """Backtest model on historical data"""
    split_idx = int(len(data) * train_size)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    all_predictions = []
    actual_values = []
    
    # Walk-forward validation
    for i in range(0, len(test_data) - forecast_days, forecast_days):
        # Train on expanding window
        current_train = pd.concat([train_data, test_data.iloc[:i]])
        
        # Make prediction
        if hasattr(model, 'build_model'):
            model.build_model(current_train)
        if hasattr(model, 'train'):
            # For LSTM, we need to prepare data differently
            X, y = model.prepare_data(current_train)
            model.build_model()
            model.train(X, y, epochs=20)
        
        # Predict
        predictions = model.predict(current_train)
        
        # Store results
        if isinstance(predictions, list):
            pred_values = [p['predicted'] if isinstance(p, dict) else p for p in predictions]
        else:
            pred_values = predictions
        
        actual_window = test_data['Close'].iloc[i:i+forecast_days].values
        
        # Ensure same length
        min_len = min(len(pred_values), len(actual_window))
        all_predictions.extend(pred_values[:min_len])
        actual_values.extend(actual_window[:min_len])
    
    # Calculate metrics
    if len(all_predictions) > 0 and len(actual_values) > 0:
        metrics = calculate_metrics(actual_values, all_predictions)
    else:
        metrics = {}
    
    return metrics, all_predictions, actual_values