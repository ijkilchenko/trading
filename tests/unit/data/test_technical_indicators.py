#!/usr/bin/env python
"""
Unit tests for technical indicators module.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.technical_indicators import (
    calculate_sma, calculate_ema, calculate_rsi, 
    calculate_macd, calculate_bollinger_bands, 
    calculate_atr, calculate_adx, add_indicators
)

class TestTechnicalIndicators(unittest.TestCase):
    """Test suite for technical indicators functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
        
        # Create price series with some pattern
        close_prices = np.linspace(100, 200, 100) + np.sin(np.linspace(0, 10, 100)) * 20
        
        # Add some noise
        close_prices += np.random.normal(0, 5, 100)
        
        # Create other price data
        open_prices = close_prices - np.random.uniform(0, 5, 100)
        high_prices = np.maximum(close_prices, open_prices) + np.random.uniform(0, 10, 100)
        low_prices = np.minimum(close_prices, open_prices) - np.random.uniform(0, 10, 100)
        
        # Create volume data
        volume = np.random.uniform(1000, 5000, 100)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    def test_calculate_sma(self):
        """Test SMA calculation."""
        window = 10
        df_with_sma = calculate_sma(self.df.copy(), window)
        
        # Check if SMA column exists
        self.assertIn(f'sma_{window}', df_with_sma.columns)
        
        # Check if SMA values are correct
        for i in range(len(self.df)):
            if i >= window - 1:
                expected_sma = self.df['close'][i-window+1:i+1].mean()
                self.assertAlmostEqual(df_with_sma[f'sma_{window}'][i], expected_sma, delta=0.0001)
            else:
                self.assertTrue(np.isnan(df_with_sma[f'sma_{window}'][i]))
    
    def test_calculate_ema(self):
        """Test EMA calculation."""
        window = 10
        df_with_ema = calculate_ema(self.df.copy(), window)
        
        # Check if EMA column exists
        self.assertIn(f'ema_{window}', df_with_ema.columns)
        
        # Check if EMA has the right length
        self.assertEqual(len(df_with_ema[f'ema_{window}']), len(self.df))
        
        # Check if EMA has NaN values at the beginning
        self.assertTrue(df_with_ema[f'ema_{window}'][:window-1].isna().all())
        
        # Check if the rest of the values are not NaN
        self.assertTrue(df_with_ema[f'ema_{window}'][window-1:].notna().all())
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        window = 14
        df_with_rsi = calculate_rsi(self.df.copy(), window)
        
        # Check if RSI column exists
        self.assertIn(f'rsi_{window}', df_with_rsi.columns)
        
        # Check if RSI has the right length
        self.assertEqual(len(df_with_rsi[f'rsi_{window}']), len(self.df))
        
        # Check if RSI has NaN values at the beginning
        self.assertTrue(df_with_rsi[f'rsi_{window}'][:window].isna().all())
        
        # Check if RSI values are in the correct range [0, 100]
        valid_rsi = df_with_rsi[f'rsi_{window}'].dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        fast = 12
        slow = 26
        signal = 9
        df_with_macd = calculate_macd(self.df.copy(), fast, slow, signal)
        
        # Check if MACD columns exist
        self.assertIn('macd_line', df_with_macd.columns)
        self.assertIn('macd_signal', df_with_macd.columns)
        self.assertIn('macd_histogram', df_with_macd.columns)
        
        # Check if MACD has the right length
        self.assertEqual(len(df_with_macd['macd_line']), len(self.df))
        
        # Check if MACD has NaN values at the beginning
        self.assertTrue(df_with_macd['macd_line'][:slow-1].isna().all())
        
        # Check if MACD signal has more NaN values due to signal period
        self.assertTrue(df_with_macd['macd_signal'][:slow+signal-2].isna().all())
        
        # Check if the histogram is the difference between MACD and signal
        valid_idx = df_with_macd['macd_histogram'].notna()
        pd.testing.assert_series_equal(
            df_with_macd['macd_histogram'][valid_idx],
            (df_with_macd['macd_line'] - df_with_macd['macd_signal'])[valid_idx],
            check_names=False
        )
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        window = 20
        num_std = 2
        df_with_bb = calculate_bollinger_bands(self.df.copy(), window, num_std)
        
        # Check if Bollinger Bands columns exist
        self.assertIn(f'bb_middle_{window}_{num_std}', df_with_bb.columns)
        self.assertIn(f'bb_upper_{window}_{num_std}', df_with_bb.columns)
        self.assertIn(f'bb_lower_{window}_{num_std}', df_with_bb.columns)
        
        # Check if Bollinger Bands have NaN values at the beginning
        self.assertTrue(df_with_bb[f'bb_middle_{window}_{num_std}'][:window-1].isna().all())
        
        # Check if middle band is the SMA
        sma = self.df['close'].rolling(window=window).mean()
        pd.testing.assert_series_equal(
            df_with_bb[f'bb_middle_{window}_{num_std}'].dropna(),
            sma.dropna(),
            check_names=False
        )
        
        # Check if upper and lower bands are at the correct distance from the middle band
        std = self.df['close'].rolling(window=window).std()
        valid_idx = df_with_bb[f'bb_upper_{window}_{num_std}'].notna()
        
        pd.testing.assert_series_equal(
            df_with_bb[f'bb_upper_{window}_{num_std}'][valid_idx],
            (sma + num_std * std)[valid_idx],
            check_names=False
        )
        
        pd.testing.assert_series_equal(
            df_with_bb[f'bb_lower_{window}_{num_std}'][valid_idx],
            (sma - num_std * std)[valid_idx],
            check_names=False
        )
    
    def test_calculate_atr(self):
        """Test ATR calculation."""
        window = 14
        df_with_atr = calculate_atr(self.df.copy(), window)
        
        # Check if ATR column exists
        self.assertIn(f'atr_{window}', df_with_atr.columns)
        
        # Check if ATR has the right length
        self.assertEqual(len(df_with_atr[f'atr_{window}']), len(self.df))
        
        # Check if ATR has NaN values at the beginning
        self.assertTrue(df_with_atr[f'atr_{window}'][:window].isna().all())
        
        # Check if ATR values are positive
        valid_atr = df_with_atr[f'atr_{window}'].dropna()
        self.assertTrue((valid_atr >= 0).all())
    
    def test_add_indicators(self):
        """Test adding multiple indicators."""
        config = {
            'indicators': [
                {'name': 'SMA', 'params': [{'window': 10}, {'window': 20}]},
                {'name': 'RSI', 'params': [{'window': 14}]},
                {'name': 'MACD', 'params': [{'fast': 12, 'slow': 26, 'signal': 9}]},
                {'name': 'BBANDS', 'params': [{'window': 20, 'num_std': 2}]},
            ]
        }
        
        df_with_indicators = add_indicators(self.df.copy(), config)
        
        # Check if all indicator columns exist
        self.assertIn('sma_10', df_with_indicators.columns)
        self.assertIn('sma_20', df_with_indicators.columns)
        self.assertIn('rsi_14', df_with_indicators.columns)
        self.assertIn('macd_line', df_with_indicators.columns)
        self.assertIn('macd_signal', df_with_indicators.columns)
        self.assertIn('macd_histogram', df_with_indicators.columns)
        self.assertIn('bb_middle_20_2', df_with_indicators.columns)
        self.assertIn('bb_upper_20_2', df_with_indicators.columns)
        self.assertIn('bb_lower_20_2', df_with_indicators.columns)

if __name__ == '__main__':
    unittest.main()
