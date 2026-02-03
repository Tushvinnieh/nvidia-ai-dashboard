import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    def __init__(self, forecast_days=7):
        self.forecast = forecast_days
        self.model = None
        self.order = None
        
    def find_best_order(self, data):
        """Find best ARIMA order using auto_arima"""
        series = data['Close']
        stepwise_model = auto_arima(
            series,
            start_p=1, start_q=1,
            max_p=3, max_q=3, m=7,
            start_P=0, seasonal=True,
            d=None, D=None, trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        self.order = stepwise_model.order
        self.seasonal_order = stepwise_model.seasonal_order
        return stepwise_model.order
    
    def build_model(self, data):
        """Build and train ARIMA model"""
        series = data['Close']
        
        # Find best parameters if not already found
        if self.order is None:
            self.find_best_order(data)
        
        # Build model
        self.model = ARIMA(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order
        )
        
        # Fit model
        self.fitted_model = self.model.fit()
        
        return self.fitted_model
    
    def predict(self, data):
        """Make predictions with confidence intervals"""
        if self.fitted_model is None:
            self.build_model(data)
        
        # Get forecast with confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=self.forecast)
        forecast_mean = forecast_result.predicted_mean
        confidence_int = forecast_result.conf_int()
        
        predictions = []
        for i in range(self.forecast):
            predictions.append({
                'predicted': forecast_mean.iloc[i],
                'lower': confidence_int.iloc[i, 0],
                'upper': confidence_int.iloc[i, 1]
            })
        
        return predictions