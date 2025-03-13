#!/usr/bin/env python
"""
Unit tests for backtester module.
"""
import os
import sys
import unittest
import tempfile
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backtesting.backtester import Backtester
from strategies.base_strategy import BaseStrategy

class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name="MockStrategy"):
        """Initialize the strategy."""
        super().__init__(name)
    
    def generate_signals(self, data):
        """Generate mock signals based on a simple pattern."""
        # Generate signals that alternate every 5 periods
        signals = np.zeros(len(data))
        
        for i in range(len(data)):
            period = (i // 5) % 3
            
            if period == 0:
                signals[i] = 1  # Buy signal
            elif period == 1:
                signals[i] = 0  # Hold signal
            else:
                signals[i] = -1  # Sell signal
        
        return pd.Series(signals, index=data.index)

class TestBacktester(unittest.TestCase):
    """Test suite for backtester functionality."""
    
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
  stop_loss: 0.05
  take_profit: 0.1
  trailing_stop: 0.03
  max_open_positions: 3
  metrics:
    - total_return
    - sharpe_ratio
    - max_drawdown
    - win_rate
  output_dir: {test_dir}/backtest_results
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
        self.backtester = Backtester(self.config_path)
        self.strategy = MockStrategy()
    
    def test_initialization(self):
        """Test backtester initialization."""
        # Check if backtester was initialized correctly
        self.assertEqual(self.backtester.initial_capital, 10000.0)
        self.assertEqual(self.backtester.position_sizing, 'fixed')
        self.assertEqual(self.backtester.fixed_position_size, 1000.0)
        self.assertEqual(self.backtester.fees['maker'], 0.001)
        self.assertEqual(self.backtester.fees['taker'], 0.001)
        self.assertEqual(self.backtester.slippage, 0.001)
    
    def test_run_backtest(self):
        """Test running a backtest."""
        # Run backtest
        results = self.backtester.run_backtest(
            self.strategy, self.test_data, 'BTCUSDT'
        )
        
        # Check if results contain all expected metrics
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('win_rate', results)
        
        # Check if trade history was recorded
        self.assertIn('trades', results)
        self.assertGreater(len(results['trades']), 0)
        
        # Check if equity curve was generated
        self.assertIn('equity_curve', results)
        self.assertEqual(len(results['equity_curve']), len(self.test_data))
    
    def test_calculate_position_size(self):
        """Test position size calculation methods."""
        # Test fixed position sizing
        self.backtester.position_sizing = 'fixed'
        position_size = self.backtester._calculate_position_size(
            'BTCUSDT', self.test_data.iloc[10]['close'], 1
        )
        self.assertEqual(position_size, self.backtester.fixed_position_size)
        
        # Test percentage position sizing
        self.backtester.position_sizing = 'percentage'
        self.backtester.percentage = 0.1
        position_size = self.backtester._calculate_position_size(
            'BTCUSDT', self.test_data.iloc[10]['close'], 1
        )
        self.assertEqual(position_size, self.backtester.capital * self.backtester.percentage)
        
        # Test risk-based position sizing
        self.backtester.position_sizing = 'risk_based'
        self.backtester.risk_per_trade = 0.02
        stop_loss = self.test_data.iloc[10]['close'] * (1 - self.backtester.stop_loss)
        
        position_size = self.backtester._calculate_position_size(
            'BTCUSDT', self.test_data.iloc[10]['close'], 1, stop_loss
        )
        
        risk_amount = self.backtester.capital * self.backtester.risk_per_trade
        price_diff = self.test_data.iloc[10]['close'] - stop_loss
        units = risk_amount / price_diff
        expected_size = units * self.test_data.iloc[10]['close']
        
        self.assertEqual(position_size, expected_size)
    
    def test_open_position(self):
        """Test opening a position."""
        # Reset backtester
        self.backtester.reset()
        
        # Open a long position
        timestamp = self.test_data.index[10]
        price = self.test_data.iloc[10]['close']
        
        self.backtester._open_position(
            'BTCUSDT', timestamp, price, 'long'
        )
        
        # Check if position was added to open positions
        self.assertEqual(len(self.backtester.open_positions), 1)
        self.assertIn('BTCUSDT', self.backtester.open_positions)
        
        position = self.backtester.open_positions['BTCUSDT']
        
        # Check position details
        self.assertEqual(position['symbol'], 'BTCUSDT')
        self.assertEqual(position['entry_time'], timestamp)
        self.assertEqual(position['entry_price'], price)
        self.assertEqual(position['direction'], 'long')
        
        # Check capital adjustment
        expected_position_value = self.backtester.fixed_position_size
        expected_commission = expected_position_value * self.backtester.fees['taker']
        expected_capital = self.backtester.initial_capital - expected_position_value - expected_commission
        
        self.assertEqual(self.backtester.capital, expected_capital)
    
    def test_close_position(self):
        """Test closing a position."""
        # Reset backtester
        self.backtester.reset()
        
        # Open a position first
        entry_timestamp = self.test_data.index[10]
        entry_price = self.test_data.iloc[10]['close']
        
        self.backtester._open_position(
            'BTCUSDT', entry_timestamp, entry_price, 'long'
        )
        
        # Position size should be fixed
        position_size = self.backtester.fixed_position_size
        position_quantity = position_size / entry_price
        
        # Close the position with profit
        exit_timestamp = self.test_data.index[15]
        exit_price = self.test_data.iloc[15]['close']
        
        trade = self.backtester._close_position(
            'BTCUSDT', exit_timestamp, exit_price
        )
        
        # Check if position was removed from open positions
        self.assertEqual(len(self.backtester.open_positions), 0)
        
        # Check if trade was recorded correctly
        self.assertEqual(trade['symbol'], 'BTCUSDT')
        self.assertEqual(trade['entry_time'], entry_timestamp)
        self.assertEqual(trade['exit_time'], exit_timestamp)
        self.assertEqual(trade['entry_price'], entry_price)
        self.assertEqual(trade['exit_price'], exit_price)
        self.assertEqual(trade['direction'], 'long')
        
        # Check P&L calculation
        expected_pnl_pct = (exit_price / entry_price) - 1
        expected_pnl = position_size * expected_pnl_pct
        
        # Adjust for fees
        expected_entry_commission = position_size * self.backtester.fees['taker']
        expected_exit_commission = (position_quantity * exit_price) * self.backtester.fees['taker']
        expected_pnl -= (expected_entry_commission + expected_exit_commission)
        
        self.assertAlmostEqual(trade['pnl_pct'], expected_pnl_pct, places=6)
        self.assertAlmostEqual(trade['pnl'], expected_pnl, places=6)
        
        # Check if capital was updated correctly
        expected_capital = (
            self.backtester.initial_capital - position_size - expected_entry_commission +
            (position_quantity * exit_price) - expected_exit_commission
        )
        
        self.assertAlmostEqual(self.backtester.capital, expected_capital, places=6)
    
    def test_calculate_metrics(self):
        """Test calculation of performance metrics."""
        # Generate an equity curve
        dates = self.test_data.index
        equity = [self.backtester.initial_capital]
        
        for i in range(1, len(dates)):
            # Add some random returns
            daily_return = np.random.normal(0.001, 0.01)  # Mean 0.1% daily return
            equity.append(equity[-1] * (1 + daily_return))
        
        equity_curve = pd.Series(equity, index=dates)
        
        # Generate trade list
        trades = []
        for i in range(10):
            # Random entry and exit timestamps
            entry_idx = np.random.randint(0, len(dates) - 10)
            exit_idx = entry_idx + np.random.randint(1, 10)
            
            # Random P&L
            pnl = np.random.normal(50, 200)
            
            trade = {
                'entry_time': dates[entry_idx],
                'exit_time': dates[exit_idx],
                'pnl': pnl,
                'pnl_pct': pnl / 1000  # Assuming 1000 position size
            }
            
            trades.append(trade)
        
        # Calculate metrics
        metrics = self.backtester._calculate_metrics(equity_curve, trades)
        
        # Check if all required metrics are present
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        # Check total return calculation
        expected_total_return = (equity[-1] / equity[0]) - 1
        self.assertAlmostEqual(metrics['total_return'], expected_total_return, places=6)
        
        # Check win rate calculation
        win_count = sum(1 for trade in trades if trade['pnl'] > 0)
        expected_win_rate = win_count / len(trades)
        self.assertEqual(metrics['win_rate'], expected_win_rate)
    
    def test_run_multiple_backtests(self):
        """Test running multiple backtests with parameter variations."""
        # Define parameter variations
        param_variations = {
            'stop_loss': [0.03, 0.05, 0.07],
            'take_profit': [0.06, 0.1, 0.15]
        }
        
        # Run multiple backtests
        results = self.backtester.run_multiple_backtests(
            self.strategy, self.test_data, 'BTCUSDT', param_variations
        )
        
        # Check if results have the expected structure
        self.assertEqual(len(results), len(param_variations['stop_loss']) * len(param_variations['take_profit']))
        
        # Check if all parameter combinations were tested
        param_sets = set()
        for result in results:
            param_set = (result['parameters']['stop_loss'], result['parameters']['take_profit'])
            param_sets.add(param_set)
        
        expected_param_sets = set((sl, tp) for sl in param_variations['stop_loss'] for tp in param_variations['take_profit'])
        self.assertEqual(param_sets, expected_param_sets)
        
        # Check if metrics are present for all results
        for result in results:
            self.assertIn('total_return', result['metrics'])
            self.assertIn('sharpe_ratio', result['metrics'])
            self.assertIn('max_drawdown', result['metrics'])
            self.assertIn('win_rate', result['metrics'])
    
    def test_reset(self):
        """Test resetting the backtester."""
        # Change backtester state
        self.backtester.capital = 5000.0
        self.backtester.open_positions = {'BTCUSDT': {'entry_price': 100.0}}
        self.backtester.trade_history = [{'symbol': 'BTCUSDT'}]
        self.backtester.equity_curve = pd.Series([10000.0, 10050.0])
        
        # Reset the backtester
        self.backtester.reset()
        
        # Check if everything was reset
        self.assertEqual(self.backtester.capital, self.backtester.initial_capital)
        self.assertEqual(len(self.backtester.open_positions), 0)
        self.assertEqual(len(self.backtester.trade_history), 0)
        self.assertIsNone(self.backtester.equity_curve)

if __name__ == '__main__':
    unittest.main()
