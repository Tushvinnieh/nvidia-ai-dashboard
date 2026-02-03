import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    def __init__(self, forecast_horizon=7):
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.weights = {}
        self.ensemble_model = None
        
    def add_model(self, model_name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[model_name] = model
        self.weights[model_name] = weight
        
    def calculate_dynamic_weights(self, historical_predictions, actual_values):
        """Calculate weights based on recent performance"""
        weights = {}
        
        for model_name, preds in historical_predictions.items():
            if len(preds) > 0 and len(actual_values) > 0:
                # Calculate RMSE for each model
                rmse = np.sqrt(np.mean((preds - actual_values) ** 2))
                
                # Inverse weighting: better models get higher weight
                weights[model_name] = 1.0 / (rmse + 1e-8)
            else:
                weights[model_name] = 1.0  # Default weight
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def ensemble_predict(self, individual_predictions, method='weighted_average'):
        """Combine predictions from multiple models"""
        
        if method == 'weighted_average':
            # Weighted average based on model performance
            ensemble_pred = np.zeros(self.forecast_horizon)
            total_weight = 0
            
            for model_name, pred_data in individual_predictions.items():
                weight = self.weights.get(model_name, 1.0)
                
                if 'predictions' in pred_data:
                    preds = pred_data['predictions']
                    if len(preds) == self.forecast_horizon:
                        ensemble_pred += preds * weight
                        total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            # Calculate ensemble uncertainty
            # Combine uncertainties from individual models
            ensemble_std = np.zeros(self.forecast_horizon)
            for model_name, pred_data in individual_predictions.items():
                weight = self.weights.get(model_name, 1.0)
                if 'std' in pred_data:
                    ensemble_std += pred_data['std'] * weight
            
            if total_weight > 0:
                ensemble_std /= total_weight
            
            results = {
                'predictions': ensemble_pred,
                'std': ensemble_std,
                'lower_bound': ensemble_pred - 1.96 * ensemble_std,
                'upper_bound': ensemble_pred + 1.96 * ensemble_std,
                'component_predictions': individual_predictions
            }
            
            return results
        
        elif method == 'stacking':
            # Use stacking regressor
            # This would require retraining on validation data
            pass
        
        elif method == 'median':
            # Use median of predictions (robust to outliers)
            all_preds = []
            for model_name, pred_data in individual_predictions.items():
                if 'predictions' in pred_data:
                    all_preds.append(pred_data['predictions'])
            
            if all_preds:
                all_preds = np.array(all_preds)
                ensemble_pred = np.median(all_preds, axis=0)
                
                # Calculate IQR for uncertainty
                q25 = np.percentile(all_preds, 25, axis=0)
                q75 = np.percentile(all_preds, 75, axis=0)
                
                results = {
                    'predictions': ensemble_pred,
                    'lower_bound': q25,
                    'upper_bound': q75,
                    'iqr': q75 - q25
                }
                
                return results
        
        return None
    
    def backtest_ensemble(self, df, models_dict, test_size=0.2):
        """Backtest ensemble performance"""
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        all_predictions = {}
        actual_values = []
        
        # Train individual models
        for model_name, model_class in models_dict.items():
            model = model_class(forecast_horizon=self.forecast_horizon)
            
            # This would need model-specific training
            # For demonstration, we'll simulate
            predictions = []
            actuals = []
            
            # Walk-forward validation
            for i in range(0, len(test_df) - self.forecast_horizon, self.forecast_horizon):
                # Simulate predictions
                current_data = pd.concat([train_df, test_df.iloc[:i]])
                
                # Get prediction (simulated)
                pred = np.random.normal(
                    test_df['Close'].iloc[i:i+self.forecast_horizon].mean(),
                    test_df['Close'].iloc[i:i+self.forecast_horizon].std() * 0.1,
                    self.forecast_horizon
                )
                
                predictions.extend(pred)
                actuals.extend(test_df['Close'].iloc[i:i+self.forecast_horizon].values)
            
            all_predictions[model_name] = np.array(predictions)
            if len(actual_values) == 0:
                actual_values = np.array(actuals)
        
        # Calculate dynamic weights
        self.weights = self.calculate_dynamic_weights(all_predictions, actual_values)
        
        # Make ensemble predictions
        ensemble_pred = np.zeros_like(actual_values)
        for model_name, preds in all_predictions.items():
            ensemble_pred += preds * self.weights[model_name]
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(actual_values, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(actual_values, ensemble_pred))
        
        # Directional accuracy
        actual_dir = np.diff(actual_values) > 0
        pred_dir = np.diff(ensemble_pred) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': dir_acc,
            'weights': self.weights,
            'ensemble_predictions': ensemble_pred,
            'actual_values': actual_values
        }
        
        return results
    
    def generate_trading_signals(self, ensemble_predictions, current_price, confidence_threshold=70):
        """Generate trading signals based on ensemble predictions"""
        
        # Calculate expected returns
        returns = (ensemble_predictions['predictions'] - current_price) / current_price * 100
        
        # Calculate signal strength
        avg_return = np.mean(returns)
        max_return = np.max(returns)
        min_return = np.min(returns)
        
        # Calculate confidence based on prediction consistency
        # If all models agree on direction, confidence is high
        model_agreement = 0
        if 'component_predictions' in ensemble_predictions:
            directions = []
            for model_preds in ensemble_predictions['component_predictions'].values():
                if 'predictions' in model_preds:
                    model_dir = (model_preds['predictions'][0] > current_price)
                    directions.append(model_dir)
            
            if directions:
                model_agreement = sum(directions) / len(directions)
                model_agreement = max(model_agreement, 1 - model_agreement)  # Agreement in either direction
        
        # Calculate overall confidence
        volatility_factor = 1 - (ensemble_predictions['std'][0] / current_price)
        confidence = min(100, max(0, (model_agreement * 100 + volatility_factor * 100) / 2))
        
        # Generate signal
        if avg_return > 2 and confidence >= confidence_threshold:
            signal = "STRONG_BUY"
            signal_strength = "HIGH"
        elif avg_return > 1 and confidence >= confidence_threshold * 0.8:
            signal = "BUY"
            signal_strength = "MEDIUM"
        elif avg_return < -2 and confidence >= confidence_threshold:
            signal = "STRONG_SELL"
            signal_strength = "HIGH"
        elif avg_return < -1 and confidence >= confidence_threshold * 0.8:
            signal = "SELL"
            signal_strength = "MEDIUM"
        else:
            signal = "HOLD"
            signal_strength = "LOW"
        
        # Risk assessment
        risk_level = "HIGH" if (max_return - min_return) > 10 else "MEDIUM" if (max_return - min_return) > 5 else "LOW"
        
        results = {
            'signal': signal,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'expected_return': avg_return,
            'risk_level': risk_level,
            'prediction_range': (min_return, max_return),
            'time_to_target': self._estimate_time_to_target(returns, target_return=5)
        }
        
        return results
    
    def _estimate_time_to_target(self, returns, target_return=5):
        """Estimate days to reach target return"""
        cumulative_returns = np.cumsum(returns)
        
        for i, cum_ret in enumerate(cumulative_returns):
            if cum_ret >= target_return:
                return i + 1
        
        # If target not reached within forecast horizon
        if len(returns) > 0:
            # Extrapolate linearly
            avg_daily_return = np.mean(returns)
            if avg_daily_return > 0:
                return int(np.ceil(target_return / avg_daily_return))
        
        return None