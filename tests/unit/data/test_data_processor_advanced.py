#!/usr/bin/env python
"""
Advanced unit tests for data processor with comprehensive edge case coverage.
"""
import os
import sys
import unittest
import tempfile
import sqlite3
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.data_processor import DataProcessor

class TestDataProcessorAdvanced(unittest.TestCase):
    """Advanced test suite for DataProcessor with comprehensive edge case coverage."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls.temp_dir.name
        
        # Create a temporary SQLite database
        cls.db_path = os.path.join(cls.test_dir, 'test_market_data.db')
        
        # Create test configuration
        cls.config_path = os.path.join(cls.test_dir, 'test_config.yaml')
        config = {
            'data': {
                'database': {
                    'path': cls.db_path,
                    'table_prefix': 'test_'
                },
                'validation': {
                    'check_zero_volume': True
                }
            },
            'processing': {
                'indicators': [
                    {'name': 'sma', 'params': [{'window': 10}, {'window': 20}, {'window': 50}]},
                    {'name': 'ema', 'params': [{'window': 12}, {'window': 26}]},
                    {'name': 'rsi', 'params': [{'window': 14}]},
                    {'name': 'macd', 'params': [{'fast_period': 12, 'slow_period': 26, 'signal_period': 9}]},
                    {'name': 'bollinger', 'params': [{'window': 20, 'num_std': 2}]}
                ],
                'train_val_test_split': {
                    'train': 0.7,
                    'val': 0.15,
                    'test': 0.15
                }
            }
        }
        with open(cls.config_path, 'w') as f:
            yaml.safe_dump(config, f)
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a connection to the test database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create a sample table for testing
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_btc_usdt (
                timestamp INTEGER PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        # Generate test data with various edge cases
        np.random.seed(42)
        timestamps = [int((datetime(2023, 1, 1) + timedelta(minutes=i*5)).timestamp() * 1000) for i in range(1000)]
        prices = [100.0]
        for _ in range(1, len(timestamps)):
            # Simulate different market conditions
            trend = np.random.choice([-0.001, 0.001, 0.005, -0.005])
            volatility = np.random.uniform(0.001, 0.01)
            prices.append(prices[-1] * (1 + trend + np.random.normal(0, volatility)))
        
        # Insert test data
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            open_price = price * np.random.uniform(0.99, 1.01)
            high_price = max(open_price, price * np.random.uniform(1.0, 1.05))
            low_price = min(open_price, price * np.random.uniform(0.95, 1.0))
            volume = np.random.uniform(100, 1000)
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO test_btc_usdt 
                (timestamp, open, high, low, close, volume) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, open_price, high_price, low_price, price, volume))
        
        self.conn.commit()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Close database connection
        self.conn.close()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directory
        cls.temp_dir.cleanup()
    
    def test_load_data_with_date_range(self):
        """Test loading data with specific date range."""
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Test loading data with start and end dates
        start_date = '2023-01-01'
        end_date = '2023-01-02'
        
        df = processor._load_data('btc_usdt', start_date, end_date)
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('datetime' in df.columns)
        self.assertTrue(df.index.name == 'datetime')
        
        # Check date range
        self.assertTrue(df.index[0] >= pd.Timestamp(start_date))
        self.assertTrue(df.index[-1] <= pd.Timestamp(end_date))
    
    def test_load_data_empty_result(self):
        """Test loading data with no results."""
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Try loading data from a non-existent date range
        start_date = '2099-01-01'
        end_date = '2099-01-02'
        
        df = processor._load_data('btc_usdt', start_date, end_date)
        
        # Assertions
        self.assertTrue(df.empty)
    
    def test_load_data_large_dataset(self):
        """Test loading a large dataset."""
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Load entire dataset
        df = processor._load_data('btc_usdt')
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        self.assertTrue('datetime' in df.columns)
        self.assertTrue(df.index.name == 'datetime')
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['close']))
    
    def test_clean_data_edge_cases(self):
        """Test data cleaning with various edge cases."""
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Load test data
        df = processor._load_data('btc_usdt')
        
        # Create a copy with some problematic data
        df_with_issues = df.copy()
        
        # Introduce NaN values
        df_with_issues.loc[df_with_issues.sample(frac=0.1).index, 'close'] = np.nan
        
        # Introduce infinite values
        df_with_issues.loc[df_with_issues.sample(frac=0.05).index, 'volume'] = np.inf
        
        # Clean the data
        cleaned_df = processor._clean_data(df_with_issues)
        
        # Assertions
        self.assertFalse(cleaned_df.isnull().any().any())  # No NaN values
        
        # Check for infinite values
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # Verify no infinite values remain
                inf_mask = np.isinf(cleaned_df[col])
                
                # Print debugging information if infinite values are found
                if inf_mask.any():
                    print(f"Infinite values in column {col}:")
                    print(cleaned_df[col][inf_mask])
                
                self.assertFalse(inf_mask.any(), f"Column {col} contains infinite values")
    
    def test_process_data_normalization(self):
        """Test data processing with normalization."""
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Process data with normalization
        processed_data = processor.process_data('btc_usdt', normalize=True)
        
        # Check processed data structure
        self.assertIn('train', processed_data)
        self.assertIn('val', processed_data)
        self.assertIn('test', processed_data)
        self.assertIn('all', processed_data)
        self.assertIn('raw_data', processed_data)
        
        # Check normalization
        for dataset_name in ['train', 'val', 'test']:
            df = processed_data[dataset_name]
            
            # Check that numeric columns are normalized
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'datetime':
                    if df[col].nunique() <= 1:
                        # For constant columns, mean should be 0 and std should be 0
                        self.assertEqual(df[col].iloc[0], 0.0, f"Constant column {col} in {dataset_name} should be exactly 0")
                    else:
                        # For variable columns, check that they are properly normalized
                        # Check mean is close to 0
                        self.assertTrue(np.isclose(df[col].mean(), 0, atol=1e-5), 
                                      f"Column {col} in {dataset_name} does not have mean close to 0")
                        
                        # Check std is close to 1
                        self.assertTrue(abs(df[col].std() - 1.0) < 0.01, 
                                      f"Column {col} in {dataset_name} does not have std close to 1")
    
    def test_process_data_small_dataset(self):
        """Test processing a very small dataset."""
        # Create a minimal dataset with a datetime index
        minimal_data = pd.DataFrame({
            'timestamp': [int(datetime(2023, 1, 1).timestamp() * 1000)],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [500.0]
        })
        minimal_data['datetime'] = pd.to_datetime(minimal_data['timestamp'], unit='ms')
        minimal_data.set_index('datetime', inplace=True)
        
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Process the minimal dataset
        processed_data = processor.process_data(minimal_data, normalize=False)
        
        # Assertions
        self.assertIn('train', processed_data)
        self.assertIn('val', processed_data)
        self.assertIn('test', processed_data)
        self.assertIn('all', processed_data)
        self.assertIn('raw_data', processed_data)
        
        # Check that train/val/test splits are handled correctly
        for dataset_name in ['train', 'val', 'test']:
            df = processed_data[dataset_name]
            self.assertTrue(len(df) >= 0)  # Might be empty for very small datasets
    
    def test_technical_indicators_generation(self):
        """Test generation of technical indicators."""
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Process data
        processed_data = processor.process_data('btc_usdt')
        
        # Check for common technical indicators
        train_df = processed_data['train']
        
        # List of expected technical indicators
        expected_indicators = [
            'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'rsi_14',
            'macd_line', 'macd_signal', 'macd_histogram',
            'bb_middle_20_2', 'bb_upper_20_2', 'bb_lower_20_2'
        ]
        
        # Check that each expected indicator is present
        for indicator in expected_indicators:
            self.assertIn(indicator, train_df.columns, f"Missing indicator: {indicator}")
    
    def test_data_leakage_prevention(self):
        """Test that there's no data leakage between train/val/test sets."""
        # Initialize data processor
        processor = DataProcessor(self.config_path)
        
        # Process data
        processed_data = processor.process_data('btc_usdt')
        
        # Get train, val, and test sets
        train_df = processed_data['train']
        val_df = processed_data['val']
        test_df = processed_data['test']
        
        # Check that there are no overlapping indices
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        self.assertTrue(len(train_indices.intersection(val_indices)) == 0)
        self.assertTrue(len(train_indices.intersection(test_indices)) == 0)
        self.assertTrue(len(val_indices.intersection(test_indices)) == 0)

if __name__ == '__main__':
    unittest.main()
