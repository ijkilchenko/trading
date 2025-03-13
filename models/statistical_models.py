#!/usr/bin/env python
"""
Statistical models for the trading system.

This module implements ARIMA, GARCH, and other statistical models.
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, name: str, params: Dict):
        """
        Initialize the ARIMA model.
        
        Args:
            name: Model name
            params: Model parameters (p, d, q)
        """
        super().__init__(name, params)
        self.p = params.get('p', 1)
        self.d = params.get('d', 1)
        self.q = params.get('q', 0)
    
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Fit the ARIMA model to training data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Dictionary with training metrics
        """
        # Default to predicting future close price if target not specified
        if not self.target_column:
            self.target_column = 'future_close_1'
        
        # Use close price as the feature
        self.feature_columns = ['close']
        
        # Get training data
        y_train = train_data[self.target_column].values
        
        # Suppress warnings during fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            try:
                # Fit the model
                self.model = ARIMA(train_data['close'].values, 
                                   order=(self.p, self.d, self.q))
                self.model_fit = self.model.fit()
                
                # Mark as fitted
                self.fitted = True
                
                # Evaluate on validation data
                val_metrics = self.evaluate(val_data)
                
                return {
                    'training_success': True,
                    'validation_metrics': val_metrics
                }
                
            except Exception as e:
                logger.error(f"Error fitting ARIMA model: {e}")
                return {
                    'training_success': False,
                    'error': str(e)
                }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the ARIMA model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Array of predictions
        """
        if not self.fitted:
            logger.error("Model not fitted yet")
            return np.array([])
        
        try:
            # Get input data
            x = data['close'].values
            
            # Initialize predictions array
            predictions = np.zeros(len(x))
            
            # For ARIMA, we need to use one-step ahead forecasts
            for i in range(len(x)):
                if i == 0:
                    # For the first prediction, use the fitted model
                    forecast = self.model_fit.forecast(steps=1)
                    predictions[i] = forecast[0]
                else:
                    # Update the model with new data point and forecast
                    history = x[:i]
                    model = ARIMA(history, order=(self.p, self.d, self.q))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=1)
                    predictions[i] = forecast[0]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {e}")
            return np.array([np.nan] * len(data))


class GARCHModel(BaseModel):
    """GARCH model for volatility forecasting."""
    
    def __init__(self, name: str, params: Dict):
        """
        Initialize the GARCH model.
        
        Args:
            name: Model name
            params: Model parameters (p, q)
        """
        super().__init__(name, params)
        self.p = params.get('p', 1)
        self.q = params.get('q', 1)
        self.mean = params.get('mean', 'constant')  # Mean model
        self.vol = params.get('vol', 'GARCH')  # Volatility model
        self.dist = params.get('dist', 'normal')  # Error distribution
    
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Fit the GARCH model to training data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Dictionary with training metrics
        """
        # Default to predicting future volatility
        if not self.target_column:
            self.target_column = 'future_return_1'
        
        # Use returns as input
        self.feature_columns = ['close']
        
        # Calculate returns if not already present
        if 'returns' not in train_data.columns:
            returns = 100 * train_data['close'].pct_change().dropna()
        else:
            returns = train_data['returns']
        
        # Suppress warnings during fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            try:
                # Create and fit the model
                self.model = arch_model(
                    returns, 
                    p=self.p, 
                    q=self.q, 
                    mean=self.mean, 
                    vol=self.vol, 
                    dist=self.dist
                )
                self.model_fit = self.model.fit(disp='off')
                
                # Mark as fitted
                self.fitted = True
                
                # Evaluate on validation data
                val_metrics = self.evaluate(val_data)
                
                return {
                    'training_success': True,
                    'validation_metrics': val_metrics
                }
                
            except Exception as e:
                logger.error(f"Error fitting GARCH model: {e}")
                return {
                    'training_success': False,
                    'error': str(e)
                }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the GARCH model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Array of predicted volatilities
        """
        if not self.fitted:
            logger.error("Model not fitted yet")
            return np.array([])
        
        try:
            # Calculate returns if not already present
            if 'returns' not in data.columns:
                returns = 100 * data['close'].pct_change().dropna()
            else:
                returns = data['returns']
            
            # Get forecasts
            forecast = self.model_fit.forecast(horizon=1)
            
            # Extract conditional volatility
            predictions = np.sqrt(forecast.variance.values[-len(returns):, 0])
            
            # Pad with NaN to match original data length if needed
            if len(predictions) < len(data):
                pad_length = len(data) - len(predictions)
                predictions = np.pad(predictions, (pad_length, 0), 
                                    mode='constant', constant_values=np.nan)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making GARCH predictions: {e}")
            return np.array([np.nan] * len(data))
