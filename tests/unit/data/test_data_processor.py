#!/usr/bin/env python
"""
Unit tests for data processor module.
"""
import os
import sys
import unittest
import tempfile
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test suite for data processor functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls.temp_dir.name
        
        # Create a test database
        cls.db_path = os.path.join(cls.test_dir, 'test_market_data.db')
        conn = sqlite3.connect(cls.db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_btcusdt (
            timestamp INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        """)
        
        # Generate test data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        delta = timedelta(minutes=1)
        
        data = []
        current_date = start_date
        price = 20000.0
        
        while current_date <= end_date:
            timestamp = int(current_date.timestamp() * 1000)
            
            # Add some sinusoidal pattern
            cycle = (current_date.hour * 60 + current_date.minute) / (24 * 60)
            price_change = np.sin(cycle * 2 * np.pi) * 200 + np.random.normal(0, 50)
            price += price_change
            
            # Ensure price is positive
            price = max(price, 100)
            
            # Generate OHLCV data
            open_price = price
            high_price = price * (1 + np.random.uniform(0, 0.01))
            low_price = price * (1 - np.random.uniform(0, 0.01))
            close_price = price * (1 + np.random.uniform(-0.005, 0.005))
            volume = np.random.uniform(1, 100)
            
            # Insert zero volume and price anomalies occasionally
            if np.random.random() < 0.01:  # 1% chance
                volume = 0
            
            if np.random.random() < 0.005:  # 0.5% chance
                close_price = close_price * 1.5  # Price spike
            
            data.append((timestamp, open_price, high_price, low_price, close_price, volume))
            
            current_date += delta
        
        # Insert data
        cursor.executemany(
            "INSERT OR REPLACE INTO test_btcusdt VALUES (?, ?, ?, ?, ?, ?)",
            data
        )
        
        # Create another symbol for testing
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_ethusdt (
            timestamp INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        """)
        
        # Generate different data for ETH
        data = []
        current_date = start_date
        price = 1500.0
        
        while current_date <= end_date:
            timestamp = int(current_date.timestamp() * 1000)
            
            # Add some pattern
            cycle = (current_date.hour * 60 + current_date.minute) / (24 * 60)
            price_change = np.sin(cycle * 2 * np.pi) * 20 + np.random.normal(0, 5)
            price += price_change
            
            # Ensure price is positive
            price = max(price, 10)
            
            # Generate OHLCV data
            open_price = price
            high_price = price * (1 + np.random.uniform(0, 0.01))
            low_price = price * (1 - np.random.uniform(0, 0.01))
            close_price = price * (1 + np.random.uniform(-0.005, 0.005))
            volume = np.random.uniform(10, 1000)
            
            data.append((timestamp, open_price, high_price, low_price, close_price, volume))
            
            current_date += delta
        
        # Insert data
        cursor.executemany(
            "INSERT OR REPLACE INTO test_ethusdt VALUES (?, ?, ?, ?, ?, ?)",
            data
        )
        
        conn.commit()
        conn.close()
        
        # Create a test config
        cls.config_path = os.path.join(cls.test_dir, 'test_config.yaml')
        with open(cls.config_path, 'w') as f:
            f.write("""
data:
  database:
    path: {db_path}
    table_prefix: test_
  validation:
    check_missing_bars: true
    check_zero_volume: true
    max_missing_bars_pct: 5
processing:
  train_val_test_split:
    train: 0.6
    val: 0.2
    test: 0.2
  indicators:
    - name: SMA
      params:
        - window: 10
        - window: 20
    - name: RSI
      params:
        - window: 14
    - name: MACD
      params:
        - fast: 12
          slow: 26
          signal: 9
    - name: BBANDS
      params:
        - window: 20
          num_std: 2
""".format(db_path=cls.db_path.replace('\\', '\\\\')))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up for each test."""
        self.data_processor = DataProcessor(self.config_path)
    
    def test_load_data(self):
        """Test loading data from database."""
        # Test loading data for BTC
        df = self.data_processor._load_data('btcusdt')
        
        # Check if data was loaded
        self.assertFalse(df.empty)
        
        # Check if dataframe has the expected columns
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime']
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check if the datetime index was set correctly
        self.assertEqual(df.index.name, 'datetime')
        
        # Check if the date filtering works
        start_date = '2023-01-05'
        end_date = '2023-01-08'
        filtered_df = self.data_processor._load_data('btcusdt', start_date, end_date)
        
        # Check if filtering worked correctly
        self.assertTrue(filtered_df.index.min().strftime('%Y-%m-%d') >= start_date)
        self.assertTrue(filtered_df.index.max().strftime('%Y-%m-%d') <= end_date)
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Load raw data
        df = self.data_processor._load_data('btcusdt')
        
        # Introduce some duplicates
        duplicate_rows = df.sample(5).copy()
        df = pd.concat([df, duplicate_rows])
        
        # Clean the data
        cleaned_df = self.data_processor._clean_data(df)
        
        # Check if duplicates were removed
        self.assertEqual(len(cleaned_df), len(cleaned_df.index.unique()))
        
        # Check if cleaned dataframe has all the expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, cleaned_df.columns)
    
    def test_split_data(self):
        """Test time series splitting functionality."""
        # Load and clean data
        df = self.data_processor._load_data('btcusdt')
        cleaned_df = self.data_processor._clean_data(df)
        
        # Split the data
        train_df, val_df, test_df = self.data_processor._split_data(cleaned_df)
        
        # Check if splits have the expected sizes
        total_size = len(cleaned_df)
        expected_train_size = int(total_size * 0.6)
        expected_val_size = int(total_size * 0.2)
        expected_test_size = total_size - expected_train_size - expected_val_size
        
        self.assertAlmostEqual(len(train_df), expected_train_size, delta=1)
        self.assertAlmostEqual(len(val_df), expected_val_size, delta=1)
        self.assertAlmostEqual(len(test_df), expected_test_size, delta=1)
        
        # Check if the splits are in the correct order (chronological)
        if not train_df.empty and not val_df.empty:
            self.assertTrue(train_df.index.max() < val_df.index.min())
        
        if not val_df.empty and not test_df.empty:
            self.assertTrue(val_df.index.max() < test_df.index.min())
    
    def test_create_features(self):
        """Test feature creation functionality."""
        # Load and clean data
        df = self.data_processor._load_data('btcusdt')
        cleaned_df = self.data_processor._clean_data(df)
        
        # Create features
        features_df = self.data_processor._create_features(cleaned_df)
        
        # Check if technical indicators were added
        expected_indicators = [
            'sma_10', 'sma_20', 'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
            'bb_middle_20_2', 'bb_upper_20_2', 'bb_lower_20_2'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, features_df.columns)
        
        # Check if time-based features were added
        expected_time_features = ['hour', 'day_of_week', 'day_of_month', 'month']
        
        for feature in expected_time_features:
            self.assertIn(feature, features_df.columns)
        
        # Check if lagged features were added
        for lag in [1, 2, 3, 5, 10, 20]:
            self.assertIn(f'close_lag_{lag}', features_df.columns)
            self.assertIn(f'volume_lag_{lag}', features_df.columns)
            self.assertIn(f'return_lag_{lag}', features_df.columns)
    
    def test_process_data(self):
        """Test the entire data processing pipeline."""
        # Process data for BTC
        result = self.data_processor.process_data('btcusdt')
        
        # Check if result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        
        expected_keys = ['train', 'val', 'test', 'all']
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], pd.DataFrame)
        
        # Check if processed data has all the expected features
        processed_df = result['all']
        expected_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_10', 'sma_20', 'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
            'bb_middle_20_2', 'bb_upper_20_2', 'bb_lower_20_2',
            'hour', 'day_of_week', 'day_of_month', 'month'
        ]
        
        for col in expected_columns:
            self.assertIn(col, processed_df.columns)

if __name__ == '__main__':
    unittest.main()
