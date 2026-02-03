import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import holidays
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class ProphetPredictor:
    def __init__(self, forecast_horizon=7, changepoint_prior_scale=0.05):
        self.forecast_horizon = forecast_horizon
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.holidays_df = None
        self.is_trained = False
        
    def prepare_holidays(self, country='US', years=None):
        """Prepare holiday effects"""
        if years is None:
            years = list(range(2016, 2026))
        
        # Get US holidays
        us_holidays = holidays.US(years=years)
        
        holidays_list = []
        for date, name in sorted(us_holidays.items()):
            holidays_list.append({
                'holiday': 'us_holiday',
                'ds': date,
                'lower_window': -2,  # 2 days before holiday
                'upper_window': 1,   # 1 day after holiday
            })
        
        # Add earnings season (NVIDIA typically reports quarterly)
        # NVIDIA fiscal quarters: Jan, Apr, Jul, Oct
        for year in years:
            for month in [1, 4, 7, 10]:
                earnings_date = pd.Timestamp(f'{year}-{month:02d}-15')  # Middle of month
                holidays_list.append({
                    'holiday': 'earnings_season',
                    'ds': earnings_date,
                    'lower_window': -5,
                    'upper_window': 5,
                })
        
        self.holidays_df = pd.DataFrame(holidays_list)
        return self.holidays_df
    
    def add_regressors(self, df):
        """Add additional regressors"""
        # Technical indicators as regressors
        df_prophet = pd.DataFrame({
            'ds': df.index,
            'y': df['Close']
        })
        
        # Add technical indicators
        df_prophet['returns'] = df['Close'].pct_change()
        df_prophet['volume'] = np.log(df['Volume'] + 1)  # Log transform
        df_prophet['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Moving averages as features
        df_prophet['sma_20'] = df['Close'].rolling(window=20).mean()
        df_prophet['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        
        # Day of week effects
        df_prophet['day_of_week'] = df.index.dayofweek
        df_prophet['is_monday'] = (df.index.dayofweek == 0).astype(int)
        df_prophet['is_friday'] = (df.index.dayofweek == 4).astype(int)
        
        # Month effects
        df_prophet['month'] = df.index.month
        
        return df_prophet.dropna()
    
    def build_model(self):
        """Build Prophet model with custom configurations"""
        model = Prophet(
            # Seasonality
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            
            # Changepoints
            changepoint_prior_scale=self.changepoint_prior_scale,
            changepoint_range=0.95,  # 95% of history can have changepoints
            
            # Uncertainty
            interval_width=0.95,
            mcmc_samples=0,  # Set to >0 for full Bayesian inference (slower)
            
            # Holidays
            holidays=self.holidays_df,
            
            # Seasonality mode
            seasonality_mode='multiplicative',
            
            # Additional regressors will be added later
        )
        
        # Add custom seasonalities
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5,
            prior_scale=0.1
        )
        
        model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=3,
            prior_scale=0.05
        )
        
        self.model = model
        return model
    
    def train(self, df, regressors=None):
        """Train Prophet model"""
        # Prepare data
        df_prophet = self.add_regressors(df)
        
        # Build model
        self.build_model()
        
        # Add regressors to model
        if regressors is None:
            regressors = ['returns', 'volume', 'high_low_spread', 'sma_20', 'is_monday']
        
        for regressor in regressors:
            if regressor in df_prophet.columns:
                self.model.add_regressor(regressor)
        
        # Fit model
        self.model.fit(df_prophet)
        self.is_trained = True
        
        return self.model
    
    def predict(self, df, future_regressors=None):
        """Make predictions with Prophet"""
        if not self.is_trained:
            self.train(df)
        
        # Prepare future dataframe
        future = self.model.make_future_dataframe(
            periods=self.forecast_horizon,
            freq='D',
            include_history=False
        )
        
        # Add future regressors
        if future_regressors is None:
            # Use last known values for regressors
            df_prophet = self.add_regressors(df)
            last_values = df_prophet.iloc[-1]
            
            for regressor in self.model.extra_regressors.keys():
                future[regressor] = last_values[regressor]
        else:
            for regressor, values in future_regressors.items():
                future[regressor] = values
        
        # Make forecast
        forecast = self.model.predict(future)
        
        # Extract relevant columns
        results = []
        for _, row in forecast.iterrows():
            results.append({
                'date': row['ds'],
                'predicted': row['yhat'],
                'lower': row['yhat_lower'],
                'upper': row['yhat_upper'],
                'trend': row['trend'],
                'weekly': row['weekly'],
                'yearly': row['yearly']
            })
        
        return results
    
    def cross_validate(self, df, initial='500 days', period='90 days', horizon='30 days'):
        """Cross-validate model performance"""
        df_prophet = self.add_regressors(df)
        
        # Add regressors to model
        self.build_model()
        for col in df_prophet.columns:
            if col not in ['ds', 'y']:
                self.model.add_regressor(col)
        
        self.model.fit(df_prophet)
        
        # Cross validation
        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel="processes"
        )
        
        # Calculate metrics
        df_p = performance_metrics(df_cv)
        
        return df_cv, df_p
    
    def plot_components(self, forecast):
        """Plot forecast components"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        fig = self.model.plot_components(forecast)
        return fig