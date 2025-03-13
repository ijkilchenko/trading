#!/usr/bin/env python
"""
Unit tests for risk management module.
"""
import os
import sys
import unittest
import tempfile
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from risk_management.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    """Test suite for risk manager functionality."""
    
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
risk_management:
  max_position_size: 0.2
  max_risk_per_trade: 0.02
  max_correlated_positions: 3
  stop_loss_pct: 0.05
  take_profit_pct: 0.1
  trailing_stop_pct: 0.03
  position_sizing: risk_based
  kelly_fraction: 0.5

backtesting:
  initial_capital: 10000.0
  position_sizing: fixed
  fixed_position_size: 1000.0
  fees:
    maker: 0.001
    taker: 0.001
  slippage: 0.001
""")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up for each test."""
        self.risk_manager = RiskManager(self.config_path)
    
    def test_initialization(self):
        """Test risk manager initialization."""
        # Check if risk manager was initialized correctly
        self.assertEqual(self.risk_manager.max_position_size, 0.2)
        self.assertEqual(self.risk_manager.max_risk_per_trade, 0.02)
        self.assertEqual(self.risk_manager.max_correlated_positions, 3)
        self.assertEqual(self.risk_manager.stop_loss_pct, 0.05)
        self.assertEqual(self.risk_manager.take_profit_pct, 0.1)
        self.assertEqual(self.risk_manager.trailing_stop_pct, 0.03)
        self.assertEqual(self.risk_manager.position_sizing_method, 'risk_based')
        
        # Check capital initialization
        self.assertEqual(self.risk_manager.initial_capital, 10000.0)
        self.assertEqual(self.risk_manager.current_capital, 10000.0)
        
        # Check if positions and trade history are empty
        self.assertEqual(len(self.risk_manager.positions), 0)
        self.assertEqual(len(self.risk_manager.trade_history), 0)
    
    def test_calculate_position_size_fixed(self):
        """Test fixed position sizing."""
        # Set position sizing method to fixed
        self.risk_manager.position_sizing_method = 'fixed'
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size('BTCUSDT', 20000.0)
        
        # Check if position size is correct (20% of capital)
        expected_size = self.risk_manager.initial_capital * self.risk_manager.max_position_size
        self.assertEqual(position_size, expected_size)
    
    def test_calculate_position_size_risk_based(self):
        """Test risk-based position sizing."""
        # Risk-based position sizing is already set in config
        
        # Calculate position size with stop loss price
        entry_price = 20000.0
        stop_loss_price = 19000.0  # 5% below entry price
        
        position_size = self.risk_manager.calculate_position_size(
            'BTCUSDT', entry_price, stop_loss_price
        )
        
        # Check if position size is correct
        risk_per_unit = abs(entry_price - stop_loss_price)
        max_risk_amount = self.risk_manager.initial_capital * self.risk_manager.max_risk_per_trade
        units = max_risk_amount / risk_per_unit
        expected_size = units * entry_price
        
        # Cap to max position size
        max_allowed = self.risk_manager.initial_capital * self.risk_manager.max_position_size
        expected_size = min(expected_size, max_allowed)
        
        self.assertEqual(position_size, expected_size)
    
    def test_calculate_position_size_kelly(self):
        """Test Kelly criterion position sizing."""
        # Set position sizing method to Kelly
        self.risk_manager.position_sizing_method = 'kelly'
        
        # Calculate position size with win rate and win/loss ratio
        position_size = self.risk_manager.calculate_position_size(
            'BTCUSDT', 20000.0, None, 0.6, 2.0
        )
        
        # Check if position size is correct using Kelly formula
        # f* = (p * b - q) / b where p = win rate, q = loss rate, b = win/loss ratio
        kelly_pct = (0.6 * 2.0 - (1 - 0.6)) / 2.0
        kelly_pct *= self.risk_manager.kelly_fraction  # Half Kelly
        expected_size = self.risk_manager.initial_capital * kelly_pct
        
        # Cap to max position size
        max_allowed = self.risk_manager.initial_capital * self.risk_manager.max_position_size
        expected_size = min(expected_size, max_allowed)
        
        self.assertEqual(position_size, expected_size)
    
    def test_open_position(self):
        """Test opening a position."""
        # Open a long position
        entry_price = 20000.0
        position_size = 2000.0
        
        position = self.risk_manager.open_position(
            'BTCUSDT', entry_price, position_size, 'long'
        )
        
        # Check if position was opened correctly
        self.assertIn('BTCUSDT', self.risk_manager.positions)
        self.assertEqual(position['entry_price'], entry_price)
        self.assertEqual(position['value'], position_size)
        self.assertEqual(position['direction'], 'long')
        
        # Check quantity calculation
        expected_quantity = position_size / entry_price
        self.assertEqual(position['quantity'], expected_quantity)
        
        # Check stop loss and take profit levels
        expected_stop_loss = entry_price * (1 - self.risk_manager.stop_loss_pct)
        expected_take_profit = entry_price * (1 + self.risk_manager.take_profit_pct)
        
        self.assertEqual(position['stop_loss'], expected_stop_loss)
        self.assertEqual(position['take_profit'], expected_take_profit)
    
    def test_close_position(self):
        """Test closing a position."""
        # Open a position first
        entry_price = 20000.0
        position_size = 2000.0
        
        self.risk_manager.open_position(
            'BTCUSDT', entry_price, position_size, 'long'
        )
        
        # Close the position with profit
        exit_price = 22000.0  # 10% profit
        
        trade = self.risk_manager.close_position(
            'BTCUSDT', exit_price, 'take_profit'
        )
        
        # Check if position was closed correctly
        self.assertNotIn('BTCUSDT', self.risk_manager.positions)
        
        # Check if trade was recorded correctly
        self.assertEqual(len(self.risk_manager.trade_history), 1)
        self.assertEqual(trade['entry_price'], entry_price)
        self.assertEqual(trade['exit_price'], exit_price)
        self.assertEqual(trade['direction'], 'long')
        self.assertEqual(trade['reason'], 'take_profit')
        
        # Check P&L calculation
        expected_pnl_pct = (exit_price / entry_price) - 1
        expected_pnl = position_size * expected_pnl_pct
        
        self.assertEqual(trade['pnl_pct'], expected_pnl_pct)
        self.assertEqual(trade['pnl'], expected_pnl)
        
        # Check if capital was updated correctly
        expected_capital = self.risk_manager.initial_capital + expected_pnl
        self.assertEqual(self.risk_manager.current_capital, expected_capital)
    
    def test_update_position(self):
        """Test updating a position and checking for stop loss/take profit."""
        # Open a long position
        entry_price = 20000.0
        position_size = 2000.0
        
        self.risk_manager.open_position(
            'BTCUSDT', entry_price, position_size, 'long'
        )
        
        # Test different price scenarios
        
        # 1. Price above entry but below take profit - should hold
        current_price = 21000.0
        action = self.risk_manager.update_position(
            'BTCUSDT', current_price, pd.Timestamp.now()
        )
        
        self.assertEqual(action['action'], 'hold')
        self.assertEqual(action['reason'], 'no_trigger')
        
        # Check if max price was updated
        self.assertEqual(self.risk_manager.positions['BTCUSDT']['max_price'], current_price)
        
        # 2. Price at take profit level - should trigger take profit
        current_price = entry_price * (1 + self.risk_manager.take_profit_pct)
        action = self.risk_manager.update_position(
            'BTCUSDT', current_price, pd.Timestamp.now()
        )
        
        self.assertEqual(action['action'], 'close')
        self.assertEqual(action['reason'], 'take_profit')
        
        # Reopen position for further tests
        self.risk_manager.close_position('BTCUSDT', current_price, 'take_profit')
        self.risk_manager.open_position(
            'BTCUSDT', entry_price, position_size, 'long'
        )
        
        # 3. Price below stop loss - should trigger stop loss
        current_price = entry_price * (1 - self.risk_manager.stop_loss_pct - 0.01)
        action = self.risk_manager.update_position(
            'BTCUSDT', current_price, pd.Timestamp.now()
        )
        
        self.assertEqual(action['action'], 'close')
        self.assertEqual(action['reason'], 'stop_loss')
        
        # Reopen position for trailing stop test
        self.risk_manager.close_position('BTCUSDT', current_price, 'stop_loss')
        self.risk_manager.open_position(
            'BTCUSDT', entry_price, position_size, 'long'
        )
        
        # 4. Price rises, then falls, should trigger trailing stop
        # First update with higher price
        high_price = entry_price * 1.1
        self.risk_manager.update_position(
            'BTCUSDT', high_price, pd.Timestamp.now()
        )
        
        # Check if trailing stop was updated
        trailing_stop = high_price * (1 - self.risk_manager.trailing_stop_pct)
        self.assertEqual(self.risk_manager.positions['BTCUSDT']['trailing_stop'], trailing_stop)
        
        # Then price falls to hit trailing stop
        current_price = trailing_stop - 1
        action = self.risk_manager.update_position(
            'BTCUSDT', current_price, pd.Timestamp.now()
        )
        
        self.assertEqual(action['action'], 'close')
        self.assertEqual(action['reason'], 'trailing_stop')
    
    def test_get_risk_exposure(self):
        """Test getting risk exposure metrics."""
        # Start with no positions
        self.risk_manager.reset()
        
        # Check initial exposure
        exposure = self.risk_manager.get_risk_exposure()
        self.assertEqual(exposure['current_capital'], self.risk_manager.initial_capital)
        self.assertEqual(exposure['total_position_value'], 0)
        self.assertEqual(exposure['exposure_pct'], 0)
        self.assertEqual(exposure['long_positions'], 0)
        self.assertEqual(exposure['short_positions'], 0)
        self.assertEqual(exposure['net_exposure'], 0)
        
        # Add a long position
        self.risk_manager.open_position(
            'BTCUSDT', 20000.0, 2000.0, 'long'
        )
        
        # Check exposure with one long position
        exposure = self.risk_manager.get_risk_exposure()
        self.assertEqual(exposure['total_position_value'], 2000.0)
        self.assertEqual(exposure['exposure_pct'], 0.2)  # 2000 / 10000
        self.assertEqual(exposure['long_positions'], 1)
        self.assertEqual(exposure['short_positions'], 0)
        self.assertEqual(exposure['net_exposure'], 0.2)  # (2000 - 0) / 10000
        
        # Add a short position
        self.risk_manager.open_position(
            'ETHUSDT', 1500.0, 1000.0, 'short'
        )
        
        # Check exposure with both long and short positions
        exposure = self.risk_manager.get_risk_exposure()
        self.assertEqual(exposure['total_position_value'], 3000.0)
        self.assertEqual(exposure['exposure_pct'], 0.3)  # 3000 / 10000
        self.assertEqual(exposure['long_positions'], 1)
        self.assertEqual(exposure['short_positions'], 1)
        self.assertEqual(exposure['net_exposure'], 0.1)  # (2000 - 1000) / 10000
    
    def test_get_trade_statistics(self):
        """Test calculating trade statistics."""
        # Start with no trades
        self.risk_manager.reset()
        
        # Check statistics with no trades
        stats = self.risk_manager.get_trade_statistics()
        self.assertEqual(stats['total_trades'], 0)
        self.assertEqual(stats['win_rate'], 0)
        
        # Add some winning and losing trades
        # Winning trade 1
        self.risk_manager.open_position('BTCUSDT', 20000.0, 2000.0, 'long')
        self.risk_manager.close_position('BTCUSDT', 22000.0, 'take_profit')
        
        # Winning trade 2
        self.risk_manager.open_position('ETHUSDT', 1500.0, 1000.0, 'short')
        self.risk_manager.close_position('ETHUSDT', 1400.0, 'take_profit')
        
        # Losing trade
        self.risk_manager.open_position('XRPUSDT', 0.5, 500.0, 'long')
        self.risk_manager.close_position('XRPUSDT', 0.45, 'stop_loss')
        
        # Check statistics with three trades
        stats = self.risk_manager.get_trade_statistics()
        
        self.assertEqual(stats['total_trades'], 3)
        self.assertAlmostEqual(stats['win_rate'], 2/3, places=4)
        
        # Check if profit metrics are calculated correctly
        winning_trades = [t for t in self.risk_manager.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.risk_manager.trade_history if t['pnl'] <= 0]
        
        expected_avg_profit = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
        expected_avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
        
        self.assertAlmostEqual(stats['avg_profit'], expected_avg_profit, places=4)
        self.assertAlmostEqual(stats['avg_loss'], expected_avg_loss, places=4)
        
        # Check profit factor
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        expected_profit_factor = total_profit / total_loss
        
        self.assertAlmostEqual(stats['profit_factor'], expected_profit_factor, places=4)
    
    def test_reset(self):
        """Test resetting the risk manager."""
        # First add some positions and trades
        self.risk_manager.open_position('BTCUSDT', 20000.0, 2000.0, 'long')
        self.risk_manager.close_position('BTCUSDT', 22000.0, 'take_profit')
        
        self.risk_manager.open_position('ETHUSDT', 1500.0, 1000.0, 'short')
        
        # Reset the risk manager
        self.risk_manager.reset()
        
        # Check if everything was reset
        self.assertEqual(self.risk_manager.current_capital, self.risk_manager.initial_capital)
        self.assertEqual(len(self.risk_manager.positions), 0)
        self.assertEqual(len(self.risk_manager.trade_history), 0)

if __name__ == '__main__':
    unittest.main()
