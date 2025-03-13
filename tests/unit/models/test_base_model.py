#!/usr/bin/env python
"""
Unit tests for base model module.
"""
import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.base_model import BaseModel

class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""
    
    def __init__(self, name="ConcreteModel", params=None):
        """Initialize the model."""
        super().__init__(name, params)
        self.is_fitted = False
    
    def fit(self, train_data, val_data=None):
        """Fit the model to training data."""
        self.is_fitted = True
        
        # Mock training metrics
        metrics = {
            'train_loss': 0.05,
            'val_loss': 0.08 if val_data is not None else None,
            'epochs': 10
        }
        
        return metrics
    
    def predict(self, data):
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Generate mock predictions
        predictions = data['close'].shift(1).fillna(method='bfill')
        
        # Add some noise
        noise = np.random.normal(0, data['close'].std() * 0.01, len(data))
        predictions = predictions + noise
        
        return predictions
    
    def evaluate(self, data, true_values=None):
        """Evaluate model performance."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        predictions = self.predict(data)
        
        if true_values is None:
            true_values = data['close']
        
        # Calculate errors
        errors = true_values - predictions
        mse = (errors ** 2).mean()
        mae = abs(errors).mean()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
        
        return metrics
    
    def _preprocess_data(self, data):
        """Preprocess data before fitting or prediction."""
        # Make a copy to avoid modifying the original
        processed_data = data.copy()
        
        # Add some features
        processed_data['returns'] = processed_data['close'].pct_change().fillna(0)
        processed_data['log_returns'] = np.log(processed_data['close'] / processed_data['close'].shift(1)).fillna(0)
        
        return processed_data

class TestBaseModel(unittest.TestCase):
    """Test suite for base model functionality."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
        
        # Create price series with some pattern
        np.random.seed(42)  # For reproducibility
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        # Create dataframe
        self.test_data = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.0, 100),
            'high': prices * np.random.uniform(1.0, 1.01, 100),
            'low': prices * np.random.uniform(0.98, 0.99, 100),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)
        
        # Split data into train and val
        self.train_data = self.test_data.iloc[:70]
        self.val_data = self.test_data.iloc[70:]
        
        # Create concrete model
        self.model = ConcreteModel()
    
    def test_initialization(self):
        """Test model initialization."""
        # Test default initialization
        self.assertEqual(self.model.name, "ConcreteModel")
        self.assertEqual(self.model.params, {})
        
        # Test initialization with parameters
        params = {'param1': 10, 'param2': 'value'}
        model = ConcreteModel(name="TestModel", params=params)
        
        self.assertEqual(model.name, "TestModel")
        self.assertEqual(model.params, params)
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        # Fit the model
        metrics = self.model.fit(self.train_data, self.val_data)
        
        # Check if the model was fitted
        self.assertTrue(self.model.is_fitted)
        
        # Check if metrics were returned
        self.assertIn('train_loss', metrics)
        self.assertIn('val_loss', metrics)
        
        # Generate predictions
        predictions = self.model.predict(self.test_data)
        
        # Check if predictions have the right length
        self.assertEqual(len(predictions), len(self.test_data))
        
        # Check if predictions are numeric
        self.assertTrue(np.issubdtype(predictions.dtype, np.number))
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Fit the model first
        self.model.fit(self.train_data)
        
        # Evaluate the model
        metrics = self.model.evaluate(self.test_data)
        
        # Check if metrics were calculated
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        
        # Check if metrics are positive
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
        
        # Check relationship between MSE and RMSE
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(metrics['mse']), places=6)
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the model without fitting (should still work)
            save_path = os.path.join(temp_dir, 'model.pkl')
            self.model.save(save_path)
            
            # Check if the file was created
            self.assertTrue(os.path.exists(save_path))
            
            # Fit the model
            self.model.fit(self.train_data)
            
            # Save the fitted model
            self.model.save(save_path)
            
            # Create a new model instance
            loaded_model = ConcreteModel()
            
            # Load the model
            loaded_model.load(save_path)
            
            # Check if the loaded model has the same attributes
            self.assertEqual(loaded_model.name, self.model.name)
            self.assertEqual(loaded_model.params, self.model.params)
            self.assertTrue(loaded_model.is_fitted)
            
            # Check if the loaded model can predict
            try:
                predictions = loaded_model.predict(self.test_data)
                self.assertEqual(len(predictions), len(self.test_data))
            except Exception as e:
                self.fail(f"Model prediction after loading failed: {e}")

    def test_get_parameters(self):
        """Test getting strategy parameters."""
        # Create strategy with parameters
        params = {'param1': 10, 'param2': 'value'}
        model = ConcreteModel(name="TestModel", params=params)
        
        # Check if parameters are correctly retrieved
        self.assertEqual(model.params, params)

    def test_set_parameters(self):
        """Test setting strategy parameters."""
        # Create strategy with initial parameters
        initial_params = {'param1': 10}
        model = ConcreteModel(name="TestModel", params=initial_params)
        
        # Set new parameters
        new_params = {'param1': 20, 'param3': 'new_value'}
        model.params = new_params
        
        # Check if parameters are correctly updated
        self.assertEqual(model.params, new_params)
    
    def test_get_name(self):
        """Test getting model name."""
        # Get name
        name = self.model.get_name()
        
        # Check if name is correct
        self.assertEqual(name, "ConcreteModel")
    
    def test_predict_without_fit(self):
        """Test prediction without fitting."""
        # Try to predict without fitting
        with self.assertRaises(ValueError):
            self.model.predict(self.test_data)
    
    def test_evaluate_without_fit(self):
        """Test evaluation without fitting."""
        # Try to evaluate without fitting
        with self.assertRaises(ValueError):
            self.model.evaluate(self.test_data)

    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Preprocess data
        processed_data = self.model._preprocess_data(self.test_data)
        
        # Check if new features were added
        self.assertIn('returns', processed_data.columns)
        self.assertIn('log_returns', processed_data.columns)
        
        # Check if original data was not modified
        self.assertNotIn('returns', self.test_data.columns)
        self.assertNotIn('log_returns', self.test_data.columns)

if __name__ == '__main__':
    unittest.main()
