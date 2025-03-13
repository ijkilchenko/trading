#!/usr/bin/env python
"""
Unit tests for strategy implementations.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from strategies.strategy_implementations import (
    MovingAverageCrossover,
    RSIStrategy,
    BollingerBandStrategy,
    MACDStrategy,
    SupportResistanceStrategy
)

class TestStrategyImplementations(unittest.TestCase):
    """Test suite for strategy implementations."""
    
    def setUp(self):
        """Set up test data for all tests."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
        
        # Create price series with some pattern
        np.random.seed(42)  # For reproducibility
        price = 100.0
        prices = []
        
        for i in range(len(dates)):
            # Add trend component
            if i < len(dates) // 3:
                trend = 0.001  # Slight uptrend
            elif i < 2 * len(dates) // 3:
                trend = 0.005  # Strong uptrend
            else:
                trend = -0.002  # Downtrend
            
            # Add cyclical component
            cycle = np.sin(i * np.pi / 10) * 0.002
            
            # Add random component
            noise = np.random.normal(0, 0.008)
            
            # Update price
            price *= (1 + trend + cycle + noise)
            prices.append(price)
        
        # Create dataframe
        self.df = pd.DataFrame({
            'open': prices * np.random.uniform(0.998, 1.002, len(prices)),
            'high': prices * np.random.uniform(1.001, 1.015, len(prices)),
            'low': prices * np.random.uniform(0.985, 0.999, len(prices)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, len(prices))
        }, index=dates)
        
        # Add technical indicators that will be used by strategies
        # SMA indicators
        self.df['sma_5'] = self.df['close'].rolling(window=5).mean()
        self.df['sma_10'] = self.df['close'].rolling(window=10).mean()
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
        
        # EMA indicators
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.df['macd_line'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=9, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['macd_signal']
        
        # Bollinger Bands
        window = 20
        std_dev = 2
        rolling_mean = self.df['close'].rolling(window=window).mean()
        rolling_std = self.df['close'].rolling(window=window).std()
        self.df['bb_middle_20_2'] = rolling_mean
        self.df['bb_upper_20_2'] = rolling_mean + (rolling_std * std_dev)
        self.df['bb_lower_20_2'] = rolling_mean - (rolling_std * std_dev)
        
        # Support and resistance levels (simplified for testing)
        self.df['support_level'] = self.df['close'].rolling(window=10).min()
        self.df['resistance_level'] = self.df['close'].rolling(window=10).max()
    
    def test_moving_average_crossover(self):
        """Test MovingAverageCrossover strategy."""
        # Create strategy with default parameters
        strategy = MovingAverageCrossover()
        
        # Generate signals
        signals = strategy.generate_signals(self.df)
        
        # Check if signals have the right length
        self.assertEqual(len(signals), len(self.df))
        
        # Check if signals have the right index
        pd.testing.assert_index_equal(signals.index, self.df.index)
        
        # Verify signal logic: Buy when fast MA crosses above slow MA
        for i in range(20, len(self.df)):  # Skip NaN values at the beginning
            if (self.df['sma_10'].iloc[i-1] <= self.df['sma_50'].iloc[i-1] and 
                self.df['sma_10'].iloc[i] > self.df['sma_50'].iloc[i]):
                self.assertEqual(signals.iloc[i], 1, f"Expected buy signal at index {i}")
            elif (self.df['sma_10'].iloc[i-1] >= self.df['sma_50'].iloc[i-1] and 
                  self.df['sma_10'].iloc[i] < self.df['sma_50'].iloc[i]):
                self.assertEqual(signals.iloc[i], -1, f"Expected sell signal at index {i}")
        
        # Test with custom parameters
        params = {'fast_ma': 'sma_5', 'slow_ma': 'sma_20'}
        custom_strategy = MovingAverageCrossover(params=params)
        
        # Generate signals
        custom_signals = custom_strategy.generate_signals(self.df)
        
        # Verify signal logic with custom parameters
        for i in range(20, len(self.df)):  # Skip NaN values at the beginning
            if (self.df['sma_5'].iloc[i-1] <= self.df['sma_20'].iloc[i-1] and 
                self.df['sma_5'].iloc[i] > self.df['sma_20'].iloc[i]):
                self.assertEqual(custom_signals.iloc[i], 1, f"Expected buy signal at index {i}")
            elif (self.df['sma_5'].iloc[i-1] >= self.df['sma_20'].iloc[i-1] and 
                  self.df['sma_5'].iloc[i] < self.df['sma_20'].iloc[i]):
                self.assertEqual(custom_signals.iloc[i], -1, f"Expected sell signal at index {i}")
    
    def test_rsi_strategy(self):
        """Test RSIStrategy."""
        # Create strategy with default parameters
        strategy = RSIStrategy()
        
        # Generate signals
        signals = strategy.generate_signals(self.df)
        
        # Check if signals have the right length
        self.assertEqual(len(signals), len(self.df))
        
        # Check signal logic: Buy when RSI crosses below oversold and then back above
        # Sell when RSI crosses above overbought and then back below
        for i in range(14, len(self.df)):  # Skip NaN values at the beginning
            if self.df['rsi_14'].iloc[i-1] < 30 and self.df['rsi_14'].iloc[i] > 30:
                self.assertEqual(signals.iloc[i], 1, f"Expected buy signal at index {i}")
            elif self.df['rsi_14'].iloc[i-1] > 70 and self.df['rsi_14'].iloc[i] < 70:
                self.assertEqual(signals.iloc[i], -1, f"Expected sell signal at index {i}")
        
        # Test with custom parameters
        params = {'rsi_period': 14, 'oversold': 25, 'overbought': 75}
        custom_strategy = RSIStrategy(params=params)
        
        # Generate signals
        custom_signals = custom_strategy.generate_signals(self.df)
        
        # Verify signal logic with custom parameters
        for i in range(14, len(self.df)):
            if self.df['rsi_14'].iloc[i-1] < 25 and self.df['rsi_14'].iloc[i] > 25:
                self.assertEqual(custom_signals.iloc[i], 1, f"Expected buy signal at index {i}")
            elif self.df['rsi_14'].iloc[i-1] > 75 and self.df['rsi_14'].iloc[i] < 75:
                self.assertEqual(custom_signals.iloc[i], -1, f"Expected sell signal at index {i}")
    
    def test_bollinger_band_strategy(self):
        """Test BollingerBandStrategy."""
        # Create strategy with default parameters
        strategy = BollingerBandStrategy()
        
        # Generate signals
        signals = strategy.generate_signals(self.df)
        
        # Check if signals have the right length
        self.assertEqual(len(signals), len(self.df))
        
        # Check signal logic: Buy when price crosses below lower band and then back above
        # Sell when price crosses above upper band and then back below
        for i in range(20, len(self.df)):  # Skip NaN values at the beginning
            if (self.df['close'].iloc[i-1] <= self.df['bb_lower_20_2'].iloc[i-1] and 
                self.df['close'].iloc[i] > self.df['bb_lower_20_2'].iloc[i]):
                self.assertEqual(signals.iloc[i], 1, f"Expected buy signal at index {i}")
            elif (self.df['close'].iloc[i-1] >= self.df['bb_upper_20_2'].iloc[i-1] and 
                  self.df['close'].iloc[i] < self.df['bb_upper_20_2'].iloc[i]):
                self.assertEqual(signals.iloc[i], -1, f"Expected sell signal at index {i}")
        
        # Test with custom parameters
        params = {'window': 20, 'num_std': 2}
        custom_strategy = BollingerBandStrategy(params=params)
        
        # Generate signals
        custom_signals = custom_strategy.generate_signals(self.df)
        
        # Custom parameters should be the same as default in this case
        pd.testing.assert_series_equal(signals, custom_signals)
    
    def test_macd_strategy(self):
        """Test MACDStrategy."""
        # Create strategy with default parameters
        strategy = MACDStrategy()
        
        # Generate signals
        signals = strategy.generate_signals(self.df)
        
        # Check if signals have the right length
        self.assertEqual(len(signals), len(self.df))
        
        # Check signal logic: Buy when MACD crosses above signal line
        # Sell when MACD crosses below signal line
        for i in range(26, len(self.df)):  # Skip NaN values at the beginning
            if (self.df['macd_line'].iloc[i-1] <= self.df['macd_signal'].iloc[i-1] and 
                self.df['macd_line'].iloc[i] > self.df['macd_signal'].iloc[i]):
                self.assertEqual(signals.iloc[i], 1, f"Expected buy signal at index {i}")
            elif (self.df['macd_line'].iloc[i-1] >= self.df['macd_signal'].iloc[i-1] and 
                  self.df['macd_line'].iloc[i] < self.df['macd_signal'].iloc[i]):
                self.assertEqual(signals.iloc[i], -1, f"Expected sell signal at index {i}")
        
        # Test with custom parameters
        params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        custom_strategy = MACDStrategy(params=params)
        
        # Generate signals
        custom_signals = custom_strategy.generate_signals(self.df)
        
        # Custom parameters should be the same as default in this case
        pd.testing.assert_series_equal(signals, custom_signals)
    
    def test_support_resistance_strategy(self):
        """Test SupportResistanceStrategy."""
        # Create strategy with default parameters
        strategy = SupportResistanceStrategy()
        
        # Generate signals
        signals = strategy.generate_signals(self.df)
        
        # Check if signals have the right length
        self.assertEqual(len(signals), len(self.df))
        
        # Check signal logic: Buy when price touches support level
        # Sell when price touches resistance level
        for i in range(10, len(self.df)):  # Skip NaN values at the beginning
            price = self.df['close'].iloc[i]
            support = self.df['support_level'].iloc[i]
            resistance = self.df['resistance_level'].iloc[i]
            
            # Define threshold for "touching" levels - within 0.5%
            support_threshold = support * 1.005
            resistance_threshold = resistance * 0.995
            
            if price <= support_threshold and price > support:
                self.assertEqual(signals.iloc[i], 1, f"Expected buy signal at index {i}")
            elif price >= resistance_threshold and price < resistance:
                self.assertEqual(signals.iloc[i], -1, f"Expected sell signal at index {i}")
    
    def test_invalid_data(self):
        """Test strategies' behavior with invalid data."""
        # Create empty dataframe
        empty_df = pd.DataFrame()
        
        # Test each strategy with empty data
        strategies = [
            MovingAverageCrossover(),
            RSIStrategy(),
            BollingerBandStrategy(),
            MACDStrategy(),
            SupportResistanceStrategy()
        ]
        
        for strategy in strategies:
            # Generate signals
            signals = strategy.generate_signals(empty_df)
            
            # Check if signals is an empty Series
            self.assertTrue(isinstance(signals, pd.Series))
            self.assertTrue(signals.empty)
        
        # Create dataframe missing required columns
        invalid_df = pd.DataFrame({'column1': [1, 2, 3]})
        
        for strategy in strategies:
            # Generate signals
            signals = strategy.generate_signals(invalid_df)
            
            # Check if signals has the right length
            self.assertEqual(len(signals), len(invalid_df))
            
            # All signals should be 0 (hold) due to missing data
            self.assertTrue((signals == 0).all())

if __name__ == '__main__':
    unittest.main()
