#!/usr/bin/env python
"""
Unit tests for grid search module.
"""
import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from strategies.grid_search import GridSearch
from strategies.base_strategy import BaseStrategy
from backtesting.backtester import Backtester

class MockStrategy(BaseStrategy):
    """Mock strategy for testing grid search."""
    
    def __init__(self, name="MockStrategy", params=None):
        """Initialize the strategy."""
        super().__init__(name, params)
    
    def generate_signals(self, data):
        """Generate mock signals based on strategy parameters."""
        signals = np.zeros(len(data))
        
        # Use parameters to influence signal generation
        threshold = self.params.get('threshold', 0.5)
        window = self.params.get('window', 10)
        
        # Generate signals based on moving average of returns
        returns = data['close'].pct_change().fillna(0)
        ma_returns = returns.rolling(window=window).mean().fillna(0)
        
        for i in range(len(data)):
            if ma_returns.iloc[i] > threshold:
                signals[i] = 1  # Buy signal
            elif ma_returns.iloc[i] < -threshold:
                signals[i] = -1  # Sell signal
        
        return pd.Series(signals, index=data.index)

class TestGridSearch(unittest.TestCase):
    """Test suite for grid search functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls.temp_dir.name
        
        # Create a test config
        cls.config_path = os.path.join(cls.test_dir, 'test_config.yaml')
        with open(cls.config_path, 'w') as f:
            f.write("""
backtesting:
  initial_capital: 10000.0
  position_sizing: fixed
  fixed_position_size: 1000.0
  fees:
    maker: 0.001
    taker: 0.001
  slippage: 0.001
  metrics:
    - total_return
    - sharpe_ratio
    - max_drawdown
    - win_rate

grid_search:
  optimization_metric: total_return
  n_jobs: 1
  output_dir: {test_dir}/grid_search_results
""".format(test_dir=cls.test_dir.replace('\\', '\\\\')))
        
        # Generate test price data
        cls.generate_test_data()
    
    @classmethod
    def generate_test_data(cls):
        """Generate test price data."""
        # Create a dataframe with test data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 1)
        dates = pd.date_range(start=start_date, end=end_date, freq='1D')
        
        # Generate price data with trend and noise
        np.random.seed(42)  # For reproducibility
        price = 100.0
        prices = []
        
        for i in range(len(dates)):
            # Add trend
            if i < len(dates) // 3:
                trend = 0.001  # Slight uptrend
            elif i < 2 * len(dates) // 3:
                trend = 0.005  # Strong uptrend
            else:
                trend = -0.002  # Downtrend
            
            # Add noise
            noise = np.random.normal(0, 0.01)
            
            # Update price
            price *= (1 + trend + noise)
            prices.append(price)
        
        # Create test dataframe
        df = pd.DataFrame({
            'open': prices * np.random.uniform(0.998, 1.002, len(prices)),
            'high': prices * np.random.uniform(1.001, 1.015, len(prices)),
            'low': prices * np.random.uniform(0.985, 0.999, len(prices)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, len(prices))
        }, index=dates)
        
        # Add some indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['rsi_14'] = 50 + np.random.normal(0, 10, len(df))  # Mock RSI
        
        # Ensure RSI is in [0, 100]
        df['rsi_14'] = df['rsi_14'].clip(0, 100)
        
        cls.test_data = df
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up for each test."""
        # Create a test config with the correct test directory
        config = {
            'backtesting': {
                'initial_capital': 10000.0,
                'position_sizing': 'fixed',
                'fixed_position_size': 1000.0,
                'fees': {
                    'maker': 0.001,
                    'taker': 0.001
                },
                'slippage': 0.001,
                'metrics': [
                    'total_return',
                    'sharpe_ratio',
                    'max_drawdown',
                    'win_rate'
                ]
            },
            'grid_search': {
                'optimization_metric': 'total_return',
                'n_jobs': 1
            },
            'outputs': {
                'base_dir': self.test_dir
            }
        }
        
        # Initialize with the config dictionary
        self.grid_search = GridSearch(config, experiment_dir=self.test_dir)
        
        # Create backtester for mocking
        self.backtester = Backtester(self.config_path)
        
        # Create mock strategy
        self.strategy = MockStrategy()
    
    def test_initialization(self):
        """Test grid search initialization."""
        # Check if grid search was initialized correctly
        self.assertEqual(self.grid_search.optimization_metric, 'total_return')
        self.assertEqual(self.grid_search.n_jobs, 1)
        self.assertEqual(self.grid_search.output_dir, os.path.join(self.test_dir, 'grid_search_results'))
    
    def test_generate_parameter_grid(self):
        """Test parameter grid generation."""
        # Define parameter ranges
        param_grid = {
            'threshold': [0.001, 0.002, 0.003],
            'window': [5, 10]
        }
        
        # Generate parameter combinations
        param_combinations = self.grid_search._generate_parameter_grid(param_grid)
        
        # Check if all combinations were generated
        self.assertEqual(len(param_combinations), 6)  # 3 thresholds x 2 windows
        
        # Check if each combination contains both parameters
        for params in param_combinations:
            self.assertIn('threshold', params)
            self.assertIn('window', params)
        
        # Check if all possible combinations are present
        expected_combinations = [
            {'threshold': 0.001, 'window': 5},
            {'threshold': 0.001, 'window': 10},
            {'threshold': 0.002, 'window': 5},
            {'threshold': 0.002, 'window': 10},
            {'threshold': 0.003, 'window': 5},
            {'threshold': 0.003, 'window': 10}
        ]
        
        for expected in expected_combinations:
            self.assertIn(expected, param_combinations)
    
    def test_evaluate_parameters(self):
        """Test parameter evaluation."""
        # Define parameters to evaluate
        params = {'threshold': 0.002, 'window': 10}
        
        # Mock backtester run_backtest method
        original_run_backtest = self.backtester.run_backtest
        
        def mock_run_backtest(strategy, data, symbol):
            return {
                'metrics': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': 0.05,
                    'win_rate': 0.6
                },
                'trades': []
            }
        
        # Patch the run_backtest method
        self.backtester.run_backtest = mock_run_backtest
        
        # Evaluate parameters
        result = self.grid_search._evaluate_parameters(
            self.strategy, params, self.test_data, 'BTCUSDT', self.backtester
        )
        
        # Check if result contains the expected metrics
        self.assertIn('total_return', result)
        self.assertIn('sharpe_ratio', result)
        self.assertEqual(result['total_return'], 0.15)
        self.assertEqual(result['sharpe_ratio'], 1.2)
        
        # Restore original method
        self.backtester.run_backtest = original_run_backtest
    
    def test_run_grid_search(self):
        """Test running a grid search."""
        # Define parameter grid
        param_grid = {
            'threshold': [0.001, 0.002],
            'window': [5, 10]
        }
        
        # Create a mock backtester that returns the expected format
        mock_backtester = MagicMock()
        
        def mock_run_backtest(strategy, data, symbol):
            params = strategy.get_parameters()
            threshold = params.get('threshold', 0)
            window = params.get('window', 0)
            
            # Generate mock results based on parameters
            total_return = 0.1 + threshold * 100 + window * 0.01
            sharpe_ratio = 1.0 + threshold * 50 + window * 0.02
            
            return {
                'metrics': {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': 0.05,
                    'win_rate': 0.6
                },
                'trades': []
            }
        
        # Set the side_effect for the mock
        mock_backtester.run_backtest.side_effect = mock_run_backtest
        
        # Run grid search with our mock
        results = self.grid_search.run_grid_search(
            self.strategy, param_grid, self.test_data, 'BTCUSDT', mock_backtester
        )
        
        # Check if results contain all parameter combinations
        self.assertEqual(len(results), 4)  # 2 thresholds x 2 windows
        
        # Check if all parameters are in the results
        self.assertTrue('threshold' in results.columns)
        self.assertTrue('window' in results.columns)
        self.assertTrue('total_return' in results.columns)
        self.assertTrue('sharpe_ratio' in results.columns)
    
    def test_save_results(self):
        """Test saving grid search results."""
        # Generate some mock results as a DataFrame
        results_data = {
            'threshold': [0.002, 0.001],
            'window': [10, 10],
            'total_return': [0.2, 0.15],
            'sharpe_ratio': [1.5, 1.2]
        }
        results_df = pd.DataFrame(results_data)
        
        # Save results
        output_file = self.grid_search.save_results(results_df, 'MockStrategy', 'BTCUSDT')
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Load results and check content
        saved_results = pd.read_csv(output_file)
        
        # Check if all results were saved
        self.assertEqual(len(saved_results), len(results_df))
        
        # Check if all columns are present
        for col in results_data.keys():
            self.assertIn(col, saved_results.columns)

if __name__ == '__main__':
    unittest.main()
