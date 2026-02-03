import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class XGBoostPredictor:
    def __init__(self, forecast_horizon=7, lookback_days=60):
        self.forecast_horizon = forecast_horizon
        self.lookback_days = lookback_days
        self.models = []  # One model for each forecast step
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_features(self, df):
        """Create extensive feature set for XGBoost"""
        features = pd.DataFrame(index=df.index)
        
        # 1. Price features
        features['close'] = df['Close']
        features['open'] = df['Open']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['volume'] = np.log1p(df['Volume'])  # Log transform
        
        # 2. Returns and volatility
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = df['Close'].pct_change().rolling(window).std()
            features[f'return_{window}d'] = df['Close'].pct_change(window)
        
        # 3. Moving averages
        for window in [5, 10, 20, 50, 200]:
            features[f'sma_{window}'] = df['Close'].rolling(window).mean()
            features[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # 4. Price position
        features['close_to_high'] = df['Close'] / df['High']
        features['close_to_low'] = df['Close'] / df['Low']
        features['high_low_ratio'] = df['High'] / df['Low']
        
        # 5. Technical indicators (custom calculations)
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        features['stochastic'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # 6. Volume indicators
        features['volume_sma'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_sma']
        
        price_volume = df['Close'] * df['Volume']
        features['vwap'] = price_volume.rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # 7. Lag features
        for lag in range(1, self.lookback_days + 1):
            if lag <= 30:  # Don't create too many lag features
                features[f'close_lag_{lag}'] = df['Close'].shift(lag)
                features[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # 8. Rolling statistics
        for window in [5, 10, 20]:
            features[f'close_roll_mean_{window}'] = df['Close'].rolling(window).mean()
            features[f'close_roll_std_{window}'] = df['Close'].rolling(window).std()
            features[f'close_roll_min_{window}'] = df['Close'].rolling(window).min()
            features[f'close_roll_max_{window}'] = df['Close'].rolling(window).max()
        
        # 9. Date features
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['week_of_year'] = df.index.isocalendar().week
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # 10. Interaction features
        features['price_volume_interaction'] = df['Close'] * np.log1p(df['Volume'])
        features['volatility_volume_interaction'] = features['volatility_20'] * features['volume_ratio']
        
        # Drop NaN values
        features = features.dropna()
        
        self.feature_names = features.columns.tolist()
        return features
    
    def prepare_data(self, features):
        """Prepare data for multi-step forecasting"""
        X, y_dict = [], {}
        
        # Create target for each forecast horizon
        for h in range(1, self.forecast_horizon + 1):
            y_dict[h] = []
        
        for i in range(self.lookback_days, len(features) - self.forecast_horizon):
            # Features: lookback window
            X.append(features.iloc[i-self.lookback_days:i].values.flatten())
            
            # Targets: future horizons
            for h in range(1, self.forecast_horizon + 1):
                y_dict[h].append(features['close'].iloc[i + h - 1])
        
        X = np.array(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare y for each horizon
        y_arrays = []
        for h in range(1, self.forecast_horizon + 1):
            y_arrays.append(np.array(y_dict[h]))
        
        return X_scaled, y_arrays
    
    def train(self, X, y_arrays, n_splits=5):
        """Train separate XGBoost model for each forecast horizon"""
        self.models = []
        
        for h in range(self.forecast_horizon):
            y = y_arrays[h]
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            best_score = float('inf')
            best_model = None
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=20,
                    eval_metric='mae'
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate
                y_pred = model.predict(X_val)
                score = mean_absolute_error(y_val, y_pred)
                
                if score < best_score:
                    best_score = score
                    best_model = model
            
            self.models.append(best_model)
            print(f"Model for horizon {h+1} trained - MAE: {best_score:.4f}")
        
        return self.models
    
    def predict(self, features):
        """Make predictions for all horizons"""
        if not self.models:
            raise ValueError("Models not trained")
        
        # Get latest data
        latest_data = features.iloc[-self.lookback_days:].values.flatten()
        latest_data_scaled = self.scaler.transform(latest_data.reshape(1, -1))
        
        # Make predictions for each horizon
        predictions = []
        std_predictions = []
        
        for h in range(self.forecast_horizon):
            # Get prediction from model for this horizon
            pred = self.models[h].predict(latest_data_scaled)[0]
            
            # Simple uncertainty estimation using training residuals
            # In production, you might want to use quantile regression or MCDropout
            predictions.append(pred)
            
            # Estimate standard deviation (simplified)
            # Use 2% of prediction as estimated std
            std_predictions.append(pred * 0.02)
        
        predictions = np.array(predictions)
        std_predictions = np.array(std_predictions)
        
        # Confidence intervals
        lower_bound = predictions - 1.96 * std_predictions
        upper_bound = predictions + 1.96 * std_predictions
        
        results = {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': std_predictions,
            'feature_importance': self.get_feature_importance()
        }
        
        return results
    
    def get_feature_importance(self, horizon=0):
        """Get feature importance for a specific horizon"""
        if not self.models:
            return {}
        
        model = self.models[horizon]
        importance = model.feature_importances_
        
        # Create feature names for flattened features
        flat_feature_names = []
        for i in range(self.lookback_days):
            for feat_name in self.feature_names:
                flat_feature_names.append(f"{feat_name}_t-{self.lookback_days - i}")
        
        # Get top 20 features
        idx = np.argsort(importance)[::-1][:20]
        top_features = {
            flat_feature_names[i]: importance[i] 
            for i in idx if importance[i] > 0
        }
        
        return top_features
    
    def save_models(self, path='xgboost_models'):
        """Save all models and scaler"""
        for i, model in enumerate(self.models):
            model.save_model(f'{path}_horizon_{i+1}.json')
        joblib.dump(self.scaler, f'{path}_scaler.pkl')
        
    def load_models(self, path='xgboost_models'):
        """Load all models and scaler"""
        self.models = []
        for i in range(self.forecast_horizon):
            model = xgb.XGBRegressor()
            model.load_model(f'{path}_horizon_{i+1}.json')
            self.models.append(model)
        
        self.scaler = joblib.load(f'{path}_scaler.pkl')