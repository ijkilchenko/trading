#!/usr/bin/env python
"""
Risk management module for the trading system.

This module handles position sizing, stop-loss, take-profit, 
and other risk management strategies to help protect trading capital.
"""
import argparse
import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management class for handling position sizing and risk controls.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the risk manager with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Extract risk management configuration
        self.risk_config = self.config.get('risk_management', {})
        
        # Default values if not specified in config
        self.max_position_size = self.risk_config.get('max_position_size', 0.2)  # 20% of capital
        self.max_risk_per_trade = self.risk_config.get('max_risk_per_trade', 0.02)  # 2% of capital
        self.max_correlated_positions = self.risk_config.get('max_correlated_positions', 3)
        self.stop_loss_pct = self.risk_config.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = self.risk_config.get('take_profit_pct', 0.1)  # 10% take profit
        self.trailing_stop_pct = self.risk_config.get('trailing_stop_pct', None)  # Optional trailing stop
        self.position_sizing_method = self.risk_config.get('position_sizing', 'fixed')
        
        # If Kelly criterion is used
        self.kelly_fraction = self.risk_config.get('kelly_fraction', 0.5)  # Half Kelly is safer
        
        # Capital allocation
        self.initial_capital = self.config.get('backtesting', {}).get('initial_capital', 10000.0)
        self.current_capital = self.initial_capital
        
        # Current positions
        self.positions = {}
        
        # Historical trades
        self.trade_history = []
    
    def calculate_position_size(self, 
                               symbol: str, 
                               entry_price: float, 
                               stop_loss_price: Optional[float] = None,
                               win_rate: Optional[float] = None,
                               avg_win_loss_ratio: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters and available capital.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price (optional)
            win_rate: Historical win rate (required for Kelly)
            avg_win_loss_ratio: Average win/loss ratio (required for Kelly)
            
        Returns:
            Position size in quote currency
        """
        # Calculate available capital
        available_capital = self._calculate_available_capital()
        
        # Calculate position size based on selected method
        if self.position_sizing_method == 'fixed':
            # Fixed percentage of available capital
            position_size = available_capital * self.max_position_size
        
        elif self.position_sizing_method == 'risk_based':
            # Position size based on risk and stop loss
            if stop_loss_price is None:
                # If no explicit stop loss provided, use the default percentage
                if entry_price > 0:
                    stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                else:
                    logger.warning(f"Invalid entry price: {entry_price}")
                    return 0.0
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit <= 0:
                logger.warning(f"Invalid risk per unit: {risk_per_unit}")
                return 0.0
            
            # Calculate units based on max risk
            max_risk_amount = available_capital * self.max_risk_per_trade
            units = max_risk_amount / risk_per_unit
            
            # Calculate position size
            position_size = units * entry_price
        
        elif self.position_sizing_method == 'kelly':
            # Kelly Criterion for position sizing
            if win_rate is None or avg_win_loss_ratio is None:
                logger.warning("Win rate and win/loss ratio required for Kelly position sizing")
                return 0.0
            
            # Kelly formula: f* = (p * b - q) / b
            # where p = win probability, q = loss probability, b = odds/win/loss ratio
            # We multiply by kelly_fraction for safety (Half Kelly)
            kelly_pct = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            kelly_pct = max(0, kelly_pct * self.kelly_fraction)  # Ensure non-negative
            
            position_size = available_capital * kelly_pct
        
        else:
            # Default to fixed position size
            position_size = available_capital * self.max_position_size
        
        # Enforce maximum position size
        max_allowed = available_capital * self.max_position_size
        position_size = min(position_size, max_allowed)
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.2f}")
        return position_size
    
    def _calculate_available_capital(self) -> float:
        """
        Calculate available capital based on current positions.
        
        Returns:
            Available capital
        """
        # Calculate total capital allocated to existing positions
        allocated_capital = sum(position['value'] for position in self.positions.values())
        
        # Calculate available capital
        available_capital = self.current_capital - allocated_capital
        
        return max(0, available_capital)
    
    def open_position(self, 
                     symbol: str, 
                     entry_price: float, 
                     position_size: float,
                     direction: str,
                     timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Open a new position.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            position_size: Position size in quote currency
            direction: 'long' or 'short'
            timestamp: Timestamp for the trade
            
        Returns:
            Position information
        """
        # Calculate quantity
        quantity = position_size / entry_price if entry_price > 0 else 0
        
        # Calculate stop loss and take profit levels
        stop_loss = entry_price * (1 - self.stop_loss_pct) if direction == 'long' else entry_price * (1 + self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct) if direction == 'long' else entry_price * (1 - self.take_profit_pct)
        
        # Position info
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': quantity,
            'direction': direction,
            'value': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': None,  # Will be updated if trailing stop is active
            'entry_time': timestamp or pd.Timestamp.now(),
            'max_price': entry_price if direction == 'long' else float('inf'),
            'min_price': entry_price if direction == 'short' else 0,
            'status': 'open'
        }
        
        # Store position
        self.positions[symbol] = position
        
        logger.info(f"Opened {direction} position for {symbol}: {quantity:.4f} units at {entry_price:.2f}")
        return position
    
    def close_position(self, 
                      symbol: str, 
                      exit_price: float, 
                      reason: str,
                      timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading pair symbol
            exit_price: Exit price
            reason: Reason for closing (e.g., 'stop_loss', 'take_profit', 'signal')
            timestamp: Timestamp for the trade
            
        Returns:
            Trade information
        """
        if symbol not in self.positions:
            logger.warning(f"No open position found for {symbol}")
            return {}
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position['direction'] == 'long':
            pnl_pct = (exit_price / position['entry_price']) - 1
        else:  # short
            pnl_pct = 1 - (exit_price / position['entry_price'])
        
        pnl = position['value'] * pnl_pct
        
        # Update capital
        self.current_capital += position['value'] + pnl
        
        # Trade info
        trade = {
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'direction': position['direction'],
            'entry_time': position['entry_time'],
            'exit_time': timestamp or pd.Timestamp.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        
        # Store trade history
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed {position['direction']} position for {symbol}: {pnl:.2f} ({pnl_pct:.2%}) due to {reason}")
        return trade
    
    def update_position(self, symbol: str, current_price: float, timestamp: pd.Timestamp) -> Dict[str, str]:
        """
        Update position with current price and check for stop loss/take profit.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Action to take ('close' or 'hold') and reason
        """
        if symbol not in self.positions:
            return {'action': 'hold', 'reason': 'no_position'}
        
        position = self.positions[symbol]
        
        # Update max/min price seen
        if position['direction'] == 'long':
            position['max_price'] = max(position['max_price'], current_price)
        else:  # short
            position['min_price'] = min(position['min_price'], current_price)
        
        # Update trailing stop if enabled
        if self.trailing_stop_pct is not None:
            if position['direction'] == 'long':
                # For long positions, trailing stop moves up with price
                trailing_stop = position['max_price'] * (1 - self.trailing_stop_pct)
                position['trailing_stop'] = trailing_stop if trailing_stop > position['stop_loss'] else position['stop_loss']
            else:  # short
                # For short positions, trailing stop moves down with price
                trailing_stop = position['min_price'] * (1 + self.trailing_stop_pct)
                position['trailing_stop'] = trailing_stop if trailing_stop < position['stop_loss'] else position['stop_loss']
        
        # Check for stop loss hit
        if position['direction'] == 'long' and current_price <= position['stop_loss']:
            return {'action': 'close', 'reason': 'stop_loss'}
        elif position['direction'] == 'short' and current_price >= position['stop_loss']:
            return {'action': 'close', 'reason': 'stop_loss'}
        
        # Check for trailing stop hit
        if position['trailing_stop'] is not None:
            if position['direction'] == 'long' and current_price <= position['trailing_stop']:
                return {'action': 'close', 'reason': 'trailing_stop'}
            elif position['direction'] == 'short' and current_price >= position['trailing_stop']:
                return {'action': 'close', 'reason': 'trailing_stop'}
        
        # Check for take profit hit
        if position['direction'] == 'long' and current_price >= position['take_profit']:
            return {'action': 'close', 'reason': 'take_profit'}
        elif position['direction'] == 'short' and current_price <= position['take_profit']:
            return {'action': 'close', 'reason': 'take_profit'}
        
        # If no conditions met, hold the position
        return {'action': 'hold', 'reason': 'no_trigger'}
    
    def get_risk_exposure(self) -> Dict[str, float]:
        """
        Get current risk exposure metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        # Calculate total position value
        total_position_value = sum(position['value'] for position in self.positions.values())
        
        # Calculate exposure as percentage of capital
        exposure_pct = total_position_value / self.current_capital if self.current_capital > 0 else 0
        
        # Count long and short positions
        long_positions = sum(1 for p in self.positions.values() if p['direction'] == 'long')
        short_positions = sum(1 for p in self.positions.values() if p['direction'] == 'short')
        
        # Calculate net exposure (long - short)
        long_value = sum(p['value'] for p in self.positions.values() if p['direction'] == 'long')
        short_value = sum(p['value'] for p in self.positions.values() if p['direction'] == 'short')
        net_exposure = (long_value - short_value) / self.current_capital if self.current_capital > 0 else 0
        
        return {
            'current_capital': self.current_capital,
            'total_position_value': total_position_value,
            'exposure_pct': exposure_pct,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'net_exposure': net_exposure
        }
    
    def get_trade_statistics(self) -> Dict[str, float]:
        """
        Calculate trading statistics based on historical trades.
        
        Returns:
            Dictionary with trading statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_win_loss_ratio': 0,
                'expectancy': 0
            }
        
        # Basic statistics
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Average profit and loss
        avg_profit = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor (total profit / total loss)
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average win/loss ratio
        avg_win_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        
        # Expectancy = (Win Rate * Average Win) - (Loss Rate * Average Loss)
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_profit) - (loss_rate * abs(avg_loss))
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'expectancy': expectancy
        }
    
    def reset(self):
        """Reset the risk manager to initial state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        
        logger.info("Risk manager reset to initial state")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Risk management module')
    
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                      help='Path to configuration file')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    setup_logger()
    
    # Initialize risk manager
    risk_manager = RiskManager(args.config)
    
    # Example usage
    logger.info("Risk Manager initialized successfully")
    
    # Print config
    logger.info(f"Max position size: {risk_manager.max_position_size}")
    logger.info(f"Max risk per trade: {risk_manager.max_risk_per_trade}")
    logger.info(f"Stop loss percentage: {risk_manager.stop_loss_pct}")
    logger.info(f"Take profit percentage: {risk_manager.take_profit_pct}")
    
    logger.info("Current risk exposure:")
    logger.info(risk_manager.get_risk_exposure())

if __name__ == "__main__":
    main()
