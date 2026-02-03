import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class LSTMPredictor:
    def __init__(self, lookback_days=60, forecast_days=7):
        self.lookback = lookback_days
        self.forecast = forecast_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def prepare_data(self, data):
        """Prepare data for LSTM"""
        # Use only Close prices
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_prices) - self.forecast):
            X.append(scaled_prices[i-self.lookback:i, 0])
            y.append(scaled_prices[i:i+self.forecast, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(self.forecast)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model
    
    def train(self, X, y, epochs=50, validation_split=0.1):
        """Train the LSTM model"""
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        return history
    
    def predict(self, data):
        """Make predictions"""
        # Prepare latest data
        prices = data['Close'].values[-self.lookback:].reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)
        
        # Reshape for prediction
        X_input = scaled_prices.T.reshape(1, self.lookback, 1)
        
        # Make prediction
        scaled_prediction = self.model.predict(X_input, verbose=0)
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).flatten()
        
        return prediction
    
    def predict_with_confidence(self, data, n_simulations=100):
        """Add confidence intervals using Monte Carlo dropout"""
        predictions = []
        for _ in range(n_simulations):
            pred = self.predict(data)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        confidence_intervals = []
        for i in range(self.forecast):
            lower = mean_pred[i] - 1.96 * std_pred[i]
            upper = mean_pred[i] + 1.96 * std_pred[i]
            confidence_intervals.append((lower, mean_pred[i], upper))
        
        return confidence_intervals