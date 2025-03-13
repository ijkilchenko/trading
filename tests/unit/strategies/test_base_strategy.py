#!/usr/bin/env python
"""
Unit tests for the base strategy module.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from strategies.base_strategy import BaseStrategy

class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""
    
    def __init__(self, name="ConcreteStrategy", params=None):
        """Initialize the strategy."""
        super().__init__(name, params)
        self.was_initialized = True
    
    def generate_signals(self, data):
        """Generate signals based on a simple pattern."""
        signals = pd.Series(np.zeros(len(data), dtype=float), index=data.index)
        
        # Set initial signal
        if self.params and 'initial_signal' in self.params:
            signals.iloc[0] = self.params['initial_signal']
        
        # Generate alternating signals every 5 bars
        for i in range(len(data)):
            if i % 5 == 0:
                signals.iloc[i] = 1.0  # Buy
            elif i % 5 == 3:
                signals.iloc[i] = -1.0  # Sell
            else:
                signals.iloc[i] = 0.0  # Hold
        
        return signals

class TestBaseStrategy(unittest.TestCase):
    """Test suite for base strategy functionality."""
    
    def setUp(self):
        """Set up test data and strategy."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='1D')
        
        # Create price data
        np.random.seed(42)  # For reproducibility
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        
        # Create dataframe
        self.test_data = pd.DataFrame({
            'open': prices * np.random.uniform(0.99, 1.0, 50),
            'high': prices * np.random.uniform(1.0, 1.01, 50),
            'low': prices * np.random.uniform(0.98, 0.99, 50),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=dates)
        
        # Add some indicators
        self.test_data['sma_10'] = self.test_data['close'].rolling(window=10).mean()
        self.test_data['sma_20'] = self.test_data['close'].rolling(window=20).mean()
        
        # Create concrete strategy
        self.strategy = ConcreteStrategy()
    
    def test_initialization(self):
        """Test strategy initialization."""
        # Test default initialization
        self.assertEqual(self.strategy.name, "ConcreteStrategy")
        self.assertEqual(self.strategy.params, {})
        self.assertTrue(self.strategy.was_initialized)
        
        # Test initialization with parameters
        params = {'param1': 10, 'param2': 'value'}
        strategy = ConcreteStrategy(name="TestStrategy", params=params)
        
        self.assertEqual(strategy.name, "TestStrategy")
        self.assertEqual(strategy.params, params)
    
    def test_generate_signals(self):
        """Test signal generation."""
        # Generate signals
        signals = self.strategy.generate_signals(self.test_data)
        
        # Check if signals have the right length
        self.assertEqual(len(signals), len(self.test_data))
        
        # Check if signals have the right index
        pd.testing.assert_index_equal(signals.index, self.test_data.index)
        
        # Check if signals have the expected pattern
        for i in range(len(signals)):
            if i % 5 == 0:
                self.assertEqual(signals.iloc[i], 1.0)  # Buy
            elif i % 5 == 3:
                self.assertEqual(signals.iloc[i], -1.0)  # Sell
            else:
                self.assertEqual(signals.iloc[i], 0.0)  # Hold
    
    def test_strategy_with_params(self):
        """Test strategy with custom parameters."""
        # Create strategy with custom parameters
        params = {'initial_signal': 1}
        strategy = ConcreteStrategy(params=params)
        
        # Generate signals
        signals = strategy.generate_signals(self.test_data)
        
        # Check if the initial signal was set correctly
        self.assertEqual(signals.iloc[0], 1)
    
    def test_get_parameters(self):
        """Test getting strategy parameters."""
        # Create strategy with parameters
        params = {'param1': 10, 'param2': 'value'}
        strategy = ConcreteStrategy(params=params)
        
        # Get parameters
        strategy_params = strategy.get_parameters()
        
        # Check if parameters are correct
        self.assertEqual(strategy_params, params)
    
    def test_set_parameters(self):
        """Test setting strategy parameters."""
        # Set new parameters
        new_params = {'param1': 20, 'param2': 'new_value'}
        self.strategy.set_parameters(new_params)
        
        # Check if parameters were updated
        self.assertEqual(self.strategy.params, new_params)
    
    def test_get_name(self):
        """Test getting strategy name."""
        # Get name
        name = self.strategy.get_name()
        
        # Check if name is correct
        self.assertEqual(name, "ConcreteStrategy")
    
    def test_invalid_data(self):
        """Test strategy behavior with invalid data."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        signals = self.strategy.generate_signals(empty_df)
        
        # Check if signals is an empty Series
        self.assertTrue(isinstance(signals, pd.Series))
        self.assertTrue(signals.empty)
        
        # Test with dataframe missing required columns
        invalid_df = pd.DataFrame({'column1': [1, 2, 3]})
        signals = self.strategy.generate_signals(invalid_df)
        
        # Check if signals has the right length
        self.assertEqual(len(signals), len(invalid_df))

if __name__ == '__main__':
    unittest.main()
