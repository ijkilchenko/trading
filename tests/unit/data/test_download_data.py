#!/usr/bin/env python
"""
Unit tests for data download module.
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

from data.download_data import BinanceDataDownloader

class TestBinanceDataDownloader(unittest.TestCase):
    """Test suite for Binance data downloader."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls.temp_dir.name
        
        # Create a test db path
        cls.db_path = os.path.join(cls.test_dir, 'test_market_data.db')
        
        # Set up test config
        cls.config = {
            'binance': {
                'base_url': 'https://api.binance.us/api/v3',
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'start_date': '2023-01-01',
                'end_date': '2023-01-07',
                'interval': '1d',
                'retries': 2,
                'timeout': 5,
                'rate_limit': 1200
            },
            'database': {
                'path': cls.db_path,
                'table_prefix': 'test_'
            },
            'validation': {
                'check_missing_bars': True,
                'check_zero_volume': True,
                'max_missing_bars_pct': 5
            }
        }
        
        # Mock sample data
        cls.mock_klines_data = [
            [
                1640995200000,  # Open time
                "46208.23000000",  # Open
                "46349.91000000",  # High
                "46128.43000000",  # Low
                "46216.93000000",  # Close
                "1043.32681000",  # Volume
                1641081599999,  # Close time
                "48243674.98998530",  # Quote asset volume
                9612,  # Number of trades
                "579.19521600",  # Taker buy base asset volume
                "26779840.98497510",  # Taker buy quote asset volume
                "0"  # Ignore
            ],
            [
                1641081600000,  # Open time
                "46216.93000000",  # Open
                "47408.78000000",  # High
                "46128.43000000",  # Low
                "47345.12000000",  # Close
                "943.30681000",  # Volume
                1641167999999,  # Close time
                "44273674.98998530",  # Quote asset volume
                8612,  # Number of trades
                "479.19521600",  # Taker buy base asset volume
                "22779840.98497510",  # Taker buy quote asset volume
                "0"  # Ignore
            ]
        ]
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up for each test."""
        # Create a fresh instance for each test
        self.downloader = BinanceDataDownloader(self.config)
    
    def test_initialization(self):
        """Test downloader initialization."""
        # Check if downloader was initialized correctly
        self.assertEqual(self.downloader.base_url, self.config['binance']['base_url'])
        self.assertEqual(self.downloader.symbols, self.config['binance']['symbols'])
        self.assertEqual(self.downloader.interval, self.config['binance']['interval'])
        self.assertEqual(self.downloader.retries, self.config['binance']['retries'])
        self.assertEqual(self.downloader.timeout, self.config['binance']['timeout'])
        self.assertEqual(self.downloader.rate_limit, self.config['binance']['rate_limit'])
        self.assertEqual(self.downloader.db_path, self.config['database']['path'])
        self.assertEqual(self.downloader.table_prefix, self.config['database']['table_prefix'])
    
    def test_parse_klines_data(self):
        """Test parsing of klines data from Binance API."""
        # Parse the mock klines data
        df = self.downloader._parse_klines_data(self.mock_klines_data)
        
        # Check if parsing was done correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.mock_klines_data))
        
        # Check column names
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertTrue(np.issubdtype(df['open'].dtype, np.number))
        self.assertTrue(np.issubdtype(df['high'].dtype, np.number))
        self.assertTrue(np.issubdtype(df['low'].dtype, np.number))
        self.assertTrue(np.issubdtype(df['close'].dtype, np.number))
        self.assertTrue(np.issubdtype(df['volume'].dtype, np.number))
        
        # Check index is timestamp
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
    
    @patch('data.download_data.requests.get')
    def test_fetch_klines(self, mock_get):
        """Test fetching klines data from Binance API."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_klines_data
        mock_get.return_value = mock_response
        
        # Fetch klines
        symbol = 'BTCUSDT'
        start_time = int(datetime(2023, 1, 1).timestamp() * 1000)
        end_time = int(datetime(2023, 1, 7).timestamp() * 1000)
        
        result = self.downloader._fetch_klines(symbol, start_time, end_time)
        
        # Check if API was called correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs['url'], f"{self.config['binance']['base_url']}/klines")
        self.assertEqual(kwargs['params']['symbol'], symbol)
        self.assertEqual(kwargs['params']['interval'], self.config['binance']['interval'])
        self.assertEqual(kwargs['params']['startTime'], start_time)
        self.assertEqual(kwargs['params']['endTime'], end_time)
        
        # Check if result is correct
        self.assertEqual(result, self.mock_klines_data)
    
    @patch('data.download_data.requests.get')
    def test_fetch_klines_error(self, mock_get):
        """Test error handling when fetching klines."""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {'code': -1100, 'msg': 'Illegal characters found in parameter'}
        mock_get.return_value = mock_response
        
        # Try to fetch klines
        symbol = 'INVALID-SYMBOL'
        start_time = int(datetime(2023, 1, 1).timestamp() * 1000)
        end_time = int(datetime(2023, 1, 7).timestamp() * 1000)
        
        with self.assertRaises(Exception):
            self.downloader._fetch_klines(symbol, start_time, end_time)
    
    @patch('data.download_data.BinanceDataDownloader._fetch_klines')
    def test_download_symbol_data(self, mock_fetch_klines):
        """Test downloading data for a specific symbol."""
        # Mock the fetch_klines method
        mock_fetch_klines.return_value = self.mock_klines_data
        
        # Download data for a symbol
        symbol = 'BTCUSDT'
        start_date = '2023-01-01'
        end_date = '2023-01-07'
        
        df = self.downloader._download_symbol_data(symbol, start_date, end_date)
        
        # Check if fetch_klines was called
        mock_fetch_klines.assert_called_once()
        
        # Check if dataframe was created correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.mock_klines_data))
    
    @patch('data.download_data.BinanceDataDownloader._download_symbol_data')
    def test_download_data(self, mock_download_symbol):
        """Test downloading data for all symbols."""
        # Mock the download_symbol_data method
        mock_df = pd.DataFrame({
            'open': [46208.23, 46216.93],
            'high': [46349.91, 47408.78],
            'low': [46128.43, 46128.43],
            'close': [46216.93, 47345.12],
            'volume': [1043.32681, 943.30681]
        }, index=pd.date_range(start='2023-01-01', periods=2))
        
        mock_download_symbol.return_value = mock_df
        
        # Download data for all symbols
        self.downloader.download_data()
        
        # Check if download_symbol_data was called for each symbol
        self.assertEqual(mock_download_symbol.call_count, len(self.config['binance']['symbols']))
    
    def test_create_db_connection(self):
        """Test database connection creation."""
        # Create a connection
        conn = self.downloader._create_db_connection()
        
        # Check if connection is valid
        self.assertIsInstance(conn, sqlite3.Connection)
        
        # Check if we can execute a query
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version();")
        version = cursor.fetchone()
        self.assertIsNotNone(version)
        
        # Close connection
        conn.close()
    
    @patch('data.download_data.BinanceDataDownloader._download_symbol_data')
    def test_save_data_to_db(self, mock_download_symbol):
        """Test saving data to database."""
        # Mock the download_symbol_data method
        mock_df = pd.DataFrame({
            'open': [46208.23, 46216.93],
            'high': [46349.91, 47408.78],
            'low': [46128.43, 46128.43],
            'close': [46216.93, 47345.12],
            'volume': [1043.32681, 943.30681]
        }, index=pd.date_range(start='2023-01-01', periods=2))
        
        mock_download_symbol.return_value = mock_df
        
        # Download and save data
        symbol = 'BTCUSDT'
        self.downloader.download_data(symbols=[symbol])
        
        # Check if data was saved to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.downloader.table_prefix}{symbol.lower()}'")
        table_exists = cursor.fetchone()
        self.assertIsNotNone(table_exists)
        
        # Check if data was inserted
        cursor.execute(f"SELECT COUNT(*) FROM {self.downloader.table_prefix}{symbol.lower()}")
        count = cursor.fetchone()[0]
        self.assertEqual(count, len(mock_df))
        
        # Close connection
        conn.close()
    
    def test_check_missing_bars(self):
        """Test checking for missing bars in data."""
        # Create a dataframe with missing bars
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1d')
        
        # Remove a few dates to simulate missing bars
        missing_dates = [dates[3], dates[5], dates[7]]
        existing_dates = [d for d in dates if d not in missing_dates]
        
        df = pd.DataFrame({
            'open': range(len(existing_dates)),
            'high': range(len(existing_dates)),
            'low': range(len(existing_dates)),
            'close': range(len(existing_dates)),
            'volume': range(len(existing_dates))
        }, index=existing_dates)
        
        # Check missing bars
        missing = self.downloader._check_missing_bars(df, '1d', dates[0], dates[-1])
        
        # Should identify the missing dates
        self.assertEqual(len(missing), len(missing_dates))
        for date in missing_dates:
            self.assertIn(date, missing)
    
    def test_validate_data(self):
        """Test data validation."""
        # Create a dataframe with some invalid data
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0],
            'high': [105.0, 106.0, 107.0, 108.0],
            'low': [95.0, 96.0, 97.0, 98.0],
            'close': [102.0, 103.0, 104.0, 105.0],
            'volume': [1000.0, 0.0, 1200.0, 1300.0]  # One zero volume
        }, index=pd.date_range(start='2023-01-01', periods=4))
        
        # Run validation
        issues = self.downloader._validate_data(df)
        
        # Should identify the zero volume issue
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("zero volume" in issue.lower() for issue in issues))

if __name__ == '__main__':
    unittest.main()
