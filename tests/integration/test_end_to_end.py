#!/usr/bin/env python
"""
Integration test for end-to-end workflow of the trading system.

This test covers the entire pipeline:
1. Data download
2. Data processing and indicator calculation
3. Model training
4. Strategy implementation
5. Backtesting
"""
import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.download_data import BinanceDataDownloader
from data.data_processor import DataProcessor
from data.technical_indicators import calculate_sma, calculate_rsi, calculate_macd
from models.train_models import train_model
from models.base_model import BaseModel
from models.statistical_models import ARIMAModel
from strategies.strategy_implementations import MovingAverageCrossover
from backtesting.backtester import Backtester

# Suppress logging during tests
logging.getLogger().setLevel(logging.ERROR)

class TestEndToEnd(unittest.TestCase):
    """Integration test for end-to-end workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls.temp_dir.name
        
        # Create a small test dataset (synthetic data)
        cls.create_test_dataset()
        
        # Create a minimal config for testing
        cls.config = {
            'data': {
                'binance': {
                    'base_url': 'https://api.binance.us/api/v3',
                    'symbols': ['BTCUSDT'],
                    'start_date': '2023-01-01',
                    'end_date': '2023-01-07',
                    'interval': '1m',
                    'retries': 2,
                    'timeout': 5,
                    'rate_limit': 1200
                },
                'database': {
                    'path': os.path.join(cls.test_dir, 'test_market_data.db'),
                    'table_prefix': 'test_'
                },
                'validation': {
                    'check_missing_bars': True,
                    'check_zero_volume': True,
                    'max_missing_bars_pct': 5
                }
            },
            'processing': {
                'train_val_test_split': {
                    'train': 0.6,
                    'val': 0.2,
                    'test': 0.2
                },
                'indicators': [
                    {'name': 'SMA', 'params': [{'window': 10}]},
                    {'name': 'RSI', 'params': [{'window': 14}]},
                    {'name': 'MACD', 'params': [{'fast': 12, 'slow': 26, 'signal': 9}]}
                ]
            },
            'models': {
                'statistical': [
                    {'name': 'ARIMA', 'params': {'p': 2, 'd': 1, 'q': 0}}
                ]
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'params': {'fast_ma': 10, 'slow_ma': 20, 'ma_type': 'sma'}
                }
            ],
            'backtesting': {
                'initial_capital': 10000.0,
                'position_sizing': 'fixed',
                'fixed_position_size': 1000.0,
                'fees': {'maker': 0.001, 'taker': 0.001},
                'slippage': 0.001,
                'output_dir': os.path.join(cls.test_dir, 'backtest_results')
            }
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.temp_dir.cleanup()
    
    @classmethod
    def create_test_dataset(cls):
        """Create a synthetic dataset for testing."""
        # Create database
        db_path = os.path.join(cls.test_dir, 'test_market_data.db')
        conn = sqlite3.connect(db_path)
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
        
        # Generate synthetic data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)
        delta = timedelta(minutes=1)
        
        data = []
        current_date = start_date
        price = 20000.0  # Starting price
        
        while current_date <= end_date:
            # Generate random price movement
            random_movement = np.random.normal(0, 100)
            price += random_movement
            
            # Ensure price is positive
            price = max(price, 100)
            
            # Generate OHLCV data
            timestamp = int(current_date.timestamp() * 1000)
            open_price = price
            high_price = price * (1 + np.random.uniform(0, 0.01))
            low_price = price * (1 - np.random.uniform(0, 0.01))
            close_price = price * (1 + np.random.uniform(-0.005, 0.005))
            volume = np.random.uniform(1, 100)
            
            data.append((timestamp, open_price, high_price, low_price, close_price, volume))
            
            current_date += delta
        
        # Insert data
        cursor.executemany(
            "INSERT OR REPLACE INTO test_btcusdt VALUES (?, ?, ?, ?, ?, ?)",
            data
        )
        
        conn.commit()
        conn.close()
        
        # Save as pickle for convenience
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save processed data
        os.makedirs(os.path.join(cls.test_dir, 'processed'), exist_ok=True)
        df.to_pickle(os.path.join(cls.test_dir, 'processed', 'test_data.pkl'))
    
    def test_data_download(self):
        """Test data download and database operations."""
        # We'll use our synthetic data instead of actually downloading from Binance
        db_path = os.path.join(self.test_dir, 'test_market_data.db')
        
        # Verify database exists and has data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM test_btcusdt")
        count = cursor.fetchone()[0]
        
        self.assertGreater(count, 0, "Database should contain test data")
        
        # Test data retrieval
        cursor.execute("SELECT * FROM test_btcusdt LIMIT 5")
        rows = cursor.fetchall()
        
        self.assertEqual(len(rows), 5, "Should retrieve 5 rows")
        self.assertEqual(len(rows[0]), 6, "Each row should have 6 columns (OHLCV + timestamp)")
        
        conn.close()
    
    def test_data_processing(self):
        """Test data processing and indicator calculation."""
        # Load test data
        df = pd.read_pickle(os.path.join(self.test_dir, 'processed', 'test_data.pkl'))
        
        # Calculate indicators
        df_with_sma = calculate_sma(df.copy(), 10)
        df_with_rsi = calculate_rsi(df.copy(), 14)
        df_with_macd = calculate_macd(df.copy(), 12, 26, 9)
        
        # Verify indicators were calculated
        self.assertIn('sma_10', df_with_sma.columns)
        self.assertIn('rsi_14', df_with_rsi.columns)
        self.assertIn('macd_line', df_with_macd.columns)
        self.assertIn('macd_signal', df_with_macd.columns)
        self.assertIn('macd_histogram', df_with_macd.columns)
        
        # Combine all indicators
        df_processed = df.copy()
        df_processed['sma_10'] = df_with_sma['sma_10']
        df_processed['sma_20'] = calculate_sma(df.copy(), 20)['sma_20']
        df_processed['rsi_14'] = df_with_rsi['rsi_14']
        df_processed['macd_line'] = df_with_macd['macd_line']
        df_processed['macd_signal'] = df_with_macd['macd_signal']
        df_processed['macd_histogram'] = df_with_macd['macd_histogram']
        
        # Save processed data for later tests
        df_processed.to_pickle(os.path.join(self.test_dir, 'processed', 'test_data_with_indicators.pkl'))
        
        # Verify data integrity
        self.assertTrue(df_processed.shape[0] > 0, "Processed data should not be empty")
        self.assertTrue(df_processed.shape[1] > 10, "Processed data should have indicators")
    
    def test_strategy_signal_generation(self):
        """Test strategy signal generation."""
        # Load processed data
        df = pd.read_pickle(os.path.join(self.test_dir, 'processed', 'test_data_with_indicators.pkl'))
        
        # Create strategy
        strategy = MovingAverageCrossover('MA_Crossover', {'fast_ma': 10, 'slow_ma': 20, 'ma_type': 'sma'})
        
        # Generate signals
        signals = strategy.generate_signals(df)
        
        # Verify signals
        self.assertEqual(len(signals), len(df), "Signal length should match data length")
        self.assertTrue(any(signals == 1), "Should have at least one buy signal")
        self.assertTrue(any(signals == -1), "Should have at least one sell signal")
    
    def test_backtesting(self):
        """Test backtesting."""
        # Load processed data
        df = pd.read_pickle(os.path.join(self.test_dir, 'processed', 'test_data_with_indicators.pkl'))
        
        # Create strategy
        strategy = MovingAverageCrossover('MA_Crossover', {'fast_ma': 10, 'slow_ma': 20, 'ma_type': 'sma'})
        
        # Generate signals
        signals = strategy.generate_signals(df)
        
        # Create backtester (with minimal config)
        backtester = Backtester({
            'backtesting': {
                'initial_capital': 10000.0,
                'position_sizing': 'fixed',
                'fixed_position_size': 1000.0,
                'fees': {'maker': 0.001, 'taker': 0.001},
                'slippage': 0.001
            }
        })
        
        # Run backtest
        metrics = backtester.backtest(df, signals)
        
        # Verify metrics
        self.assertIn('total_return', metrics, "Should calculate total return")
        self.assertIn('sharpe_ratio', metrics, "Should calculate Sharpe ratio")
        self.assertIn('max_drawdown', metrics, "Should calculate max drawdown")
        self.assertIn('win_rate', metrics, "Should calculate win rate")
    
    def test_end_to_end_pipeline(self):
        """Test the entire pipeline from data to backtesting."""
        # This test simulates the entire workflow
        
        # Step 1: Prepare data (already done in setUp)
        db_path = os.path.join(self.test_dir, 'test_market_data.db')
        self.assertTrue(os.path.exists(db_path), "Database should exist")
        
        # Step 2: Process data and calculate indicators
        df = pd.read_pickle(os.path.join(self.test_dir, 'processed', 'test_data.pkl'))
        
        processor = DataProcessor(self.config)
        df_processed = processor.process_data(df)
        
        # Step 3: Train a simple model (using a small subset for speed)
        model_params = {'p': 2, 'd': 1, 'q': 0}
        model = ARIMAModel('ARIMA', model_params)
        
        # Use a small sample for training
        train_sample = df_processed.iloc[-200:-100].copy()
        val_sample = df_processed.iloc[-100:].copy()
        
        # Define target and feature columns
        target_col = 'close'
        feature_cols = ['open', 'high', 'low', 'volume', 'sma_10', 'rsi_14']
        
        # Make sure all required features are present
        for col in feature_cols:
            if col not in train_sample.columns and col != 'sma_10' and col != 'rsi_14':
                train_sample[col] = df_processed[col]
                val_sample[col] = df_processed[col]
        
        # Add missing indicators if needed
        if 'sma_10' not in train_sample.columns:
            train_sample = calculate_sma(train_sample, 10)
            val_sample = calculate_sma(val_sample, 10)
        
        if 'rsi_14' not in train_sample.columns:
            train_sample = calculate_rsi(train_sample, 14)
            val_sample = calculate_rsi(val_sample, 14)
        
        # Train model
        train_metrics = model.fit(train_sample, val_sample)
        
        # Verify model training completed
        self.assertTrue(model.fitted, "Model should be fitted")
        
        # Step 4: Generate predictions
        predictions = model.predict(val_sample)
        
        # Verify predictions
        self.assertEqual(len(predictions), len(val_sample), "Predictions length should match validation data length")
        
        # Step 5: Generate strategy signals
        strategy = MovingAverageCrossover('MA_Crossover', {'fast_ma': 10, 'slow_ma': 20, 'ma_type': 'sma'})
        signals = strategy.generate_signals(df_processed)
        
        # Step 6: Run backtest
        backtester = Backtester(self.config)
        metrics = backtester.backtest(df_processed, signals)
        
        # Verify metrics
        self.assertIn('total_return', metrics, "Should calculate total return")
        self.assertIn('sharpe_ratio', metrics, "Should calculate Sharpe ratio")
        self.assertIn('max_drawdown', metrics, "Should calculate max drawdown")
        
        # Log success
        print(f"End-to-end test successful. Metrics: {metrics}")

if __name__ == '__main__':
    unittest.main()
