#!/usr/bin/env python
"""
Backtester module for the trading system.

This module evaluates trading strategies on historical data,
considering fees, slippage, position sizing, and other real-world trading factors.
"""
import argparse
import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class Backtester:
    """Backtester for evaluating trading strategies."""
    
    def __init__(self, config_path: str):
        """
        Initialize the backtester with configuration.
        
        Args:
            config_path: Path to the YAML configuration file or config dict
        """
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            # Handle case where config is passed directly as a dict
            self.config = config_path
        
        # Get backtesting configuration with defaults
        backtesting_config = self.config.get('backtesting', {})
        
        # Initialize parameters from config with sensible defaults
        self.initial_capital = backtesting_config.get('initial_capital', 10000.0)
        self.capital = self.initial_capital
        self.position_sizing = backtesting_config.get('position_sizing', 'fixed')
        self.fixed_position_size = backtesting_config.get('fixed_position_size', 1000.0)
        self.percentage_position_size = backtesting_config.get('percentage_position_size', 0.1)
        self.percentage = self.percentage_position_size  # For test compatibility
        self.risk_per_trade = backtesting_config.get('risk_per_trade', 0.02)
        
        # Fee configuration
        fees_config = backtesting_config.get('fees', {})
        self.fees = {
            'maker': fees_config.get('maker', 0.001),
            'taker': fees_config.get('taker', 0.001)
        }
        
        # Other parameters
        self.slippage = backtesting_config.get('slippage', 0.001)
        self.stop_loss = backtesting_config.get('stop_loss', 0.05)
        self.take_profit = backtesting_config.get('take_profit', 0.1)
        
        # Initialize tracking variables
        self.open_positions = {}
        self.trade_history = []
        self.equity_curve = None

    def reset(self):
        """
        Reset the backtester to its initial state.
        """
        self.capital = self.initial_capital
        self.open_positions = {}
        self.trade_history = []
        self.equity_curve = None

    def _calculate_position_size(self, symbol: str, price: float, units: int = 1, stop_loss: Optional[float] = None) -> float:
        """
        Calculate position size based on position sizing strategy.
        
        Args:
            symbol: Trading symbol
            price: Current price
            units: Number of units/shares
            stop_loss: Optional stop loss price for risk-based sizing
            
        Returns:
            Position size in dollars
        """
        if self.position_sizing == 'fixed':
            # Fixed dollar amount
            return self.fixed_position_size
        elif self.position_sizing == 'percentage':
            # Percentage of capital
            return self.initial_capital * self.percentage_position_size
        elif self.position_sizing == 'risk_based':
            # Risk-based position sizing
            if stop_loss is None:
                logger.warning("Stop loss not provided for risk-based position sizing. Using fixed.")
                return self.fixed_position_size
            
            # Calculate risk amount
            risk_amount = self.initial_capital * self.risk_per_trade
            
            # Calculate position size based on risk
            price_diff = price - stop_loss
            units = risk_amount / abs(price_diff)
            return units * price
        else:
            # Default to fixed
            logger.warning(f"Unknown position sizing strategy: {self.position_sizing}. Using fixed.")
            return self.fixed_position_size
    
    def _apply_slippage(self, price: float, direction: int) -> float:
        """
        Apply slippage to the execution price.
        
        Args:
            price: Original price
            direction: 1 for buy (slippage increases price), -1 for sell (slippage decreases price)
            
        Returns:
            Price with slippage applied
        """
        return price * (1 + direction * self.slippage)
    
    def _apply_fee(self, amount: float, price: float, is_maker: bool = False) -> float:
        """
        Apply trading fee.
        
        Args:
            amount: Trade amount in dollars
            price: Execution price
            is_maker: Whether the order is a maker order (limit order)
            
        Returns:
            Fee amount in dollars
        """
        fee_rate = self.fees['maker'] if is_maker else self.fees['taker']
        return amount * fee_rate
    
    def backtest(self, data: pd.DataFrame, strategy: BaseStrategy, 
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        Run backtest on historical data using a strategy.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            strategy: Trading strategy
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for strategy: {strategy.name}")
        
        # Filter data by date range if provided
        if start_date or end_date:
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize results DataFrame
        results = data[['open', 'high', 'low', 'close', 'volume']].copy()
        results['signal'] = signals
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        entry_price = 0
        equity = []
        trades = []
        
        # Loop through each day
        for i, (timestamp, row) in enumerate(results.iterrows()):
            # Current price
            price = row['close']
            
            # Current signal
            signal = row['signal']
            
            # Portfolio value at start of day (without current price update)
            portfolio_value = capital + position * price
            
            # Execute signal if not already in a position with the same direction
            if signal == 1 and position <= 0:  # Buy signal
                # Close any existing short position
                if position < 0:
                    # Calculate profit/loss on short position
                    price_with_slippage = self._apply_slippage(price, 1)  # Buy to cover
                    pnl = abs(position) * (entry_price - price_with_slippage)
                    fee = self._apply_fee(abs(position) * price_with_slippage, price_with_slippage)
                    
                    # Update capital
                    capital += pnl - fee
                    
                    # Record trade
                    trades.append({
                        'entry_date': entry_timestamp,
                        'exit_date': timestamp,
                        'entry_price': entry_price,
                        'exit_price': price_with_slippage,
                        'position': position,
                        'pnl': pnl,
                        'fee': fee,
                        'type': 'short'
                    })
                
                # Calculate new position size
                position_size = self._calculate_position_size('symbol', price)
                
                # Apply slippage for buy order
                price_with_slippage = self._apply_slippage(price, 1)
                
                # Calculate cost and fee
                cost = position_size * price_with_slippage
                fee = self._apply_fee(cost, price_with_slippage)
                
                # Update positions and capital
                position = position_size
                capital -= (cost + fee)
                entry_price = price_with_slippage
                entry_timestamp = timestamp
                
            elif signal == -1 and position >= 0:  # Sell signal
                # Close any existing long position
                if position > 0:
                    # Calculate profit/loss on long position
                    price_with_slippage = self._apply_slippage(price, -1)  # Sell to close
                    pnl = position * (price_with_slippage - entry_price)
                    fee = self._apply_fee(position * price_with_slippage, price_with_slippage)
                    
                    # Update capital
                    capital += pnl - fee
                    
                    # Record trade
                    trades.append({
                        'entry_date': entry_timestamp,
                        'exit_date': timestamp,
                        'entry_price': entry_price,
                        'exit_price': price_with_slippage,
                        'position': position,
                        'pnl': pnl,
                        'fee': fee,
                        'type': 'long'
                    })
                
                # Calculate new position size for short
                position_size = self._calculate_position_size('symbol', price)
                
                # Apply slippage for sell order
                price_with_slippage = self._apply_slippage(price, -1)
                
                # Update positions and capital
                position = -position_size
                entry_price = price_with_slippage
                entry_timestamp = timestamp
            
            # Update portfolio value at end of day
            portfolio_value = capital + position * price
            
            # Store equity curve
            equity.append({
                'date': timestamp,
                'capital': capital,
                'position_value': position * price,
                'equity': portfolio_value
            })
        
        # Close any open position at the end
        if position != 0:
            price = results['close'].iloc[-1]
            
            if position > 0:  # Long position
                price_with_slippage = self._apply_slippage(price, -1)  # Sell to close
                pnl = position * (price_with_slippage - entry_price)
                fee = self._apply_fee(position * price_with_slippage, price_with_slippage)
                
                # Update capital
                capital += pnl - fee
                
                # Record trade
                trades.append({
                    'entry_date': entry_timestamp,
                    'exit_date': results.index[-1],
                    'entry_price': entry_price,
                    'exit_price': price_with_slippage,
                    'position': position,
                    'pnl': pnl,
                    'fee': fee,
                    'type': 'long'
                })
                
            else:  # Short position
                price_with_slippage = self._apply_slippage(price, 1)  # Buy to cover
                pnl = abs(position) * (entry_price - price_with_slippage)
                fee = self._apply_fee(abs(position) * price_with_slippage, price_with_slippage)
                
                # Update capital
                capital += pnl - fee
                
                # Record trade
                trades.append({
                    'entry_date': entry_timestamp,
                    'exit_date': results.index[-1],
                    'entry_price': entry_price,
                    'exit_price': price_with_slippage,
                    'position': position,
                    'pnl': pnl,
                    'fee': fee,
                    'type': 'short'
                })
            
            # Reset position
            position = 0
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity)
        equity_df.set_index('date', inplace=True)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(equity_df, trades_df)
        
        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': metrics,
            'signals': signals
        }
    
    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            equity_curve: Series with equity values
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with performance metrics
        """
        # Convert trades list to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Reuse the existing _calculate_performance_metrics method
        metrics = self._calculate_performance_metrics(
            pd.DataFrame({'equity': equity_curve}), 
            trades_df
        )
        
        # Add max_drawdown for compatibility
        metrics['max_drawdown'] = metrics['max_drawdown_pct']
        
        return metrics

    def _calculate_performance_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            equity_df: DataFrame with equity curve
            trades_df: DataFrame with trade information
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Basic metrics
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital
        
        metrics['initial_equity'] = initial_equity
        metrics['final_equity'] = final_equity
        metrics['total_return'] = final_equity / initial_equity - 1
        metrics['total_return_pct'] = metrics['total_return'] * 100
        
        # Calculate annualized return
        if not equity_df.empty:
            days = (equity_df.index[-1] - equity_df.index[0]).days
            years = days / 365
            metrics['annualized_return'] = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0
            metrics['annualized_return_pct'] = metrics['annualized_return'] * 100
        
        # Trade metrics
        if len(trades_df) > 0:
            metrics['num_trades'] = len(trades_df)
            metrics['num_winning_trades'] = (trades_df['pnl'] > 0).sum()
            metrics['num_losing_trades'] = (trades_df['pnl'] <= 0).sum()
            metrics['win_rate'] = metrics['num_winning_trades'] / metrics['num_trades'] if metrics['num_trades'] > 0 else 0
            metrics['total_pnl'] = trades_df['pnl'].sum()
            metrics['total_fees'] = trades_df.get('fee', pd.Series(0, index=trades_df.index)).sum()
            metrics['net_pnl'] = metrics['total_pnl'] - metrics['total_fees']
            
            # Average trade metrics
            metrics['avg_trade_pnl'] = trades_df['pnl'].mean()
            metrics['avg_winning_trade'] = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean() if metrics['num_winning_trades'] > 0 else 0
            metrics['avg_losing_trade'] = trades_df.loc[trades_df['pnl'] <= 0, 'pnl'].mean() if metrics['num_losing_trades'] > 0 else 0
            
            # Profit factor
            winning_trades_sum = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum()
            losing_trades_sum = abs(trades_df.loc[trades_df['pnl'] <= 0, 'pnl'].sum())
            metrics['profit_factor'] = winning_trades_sum / losing_trades_sum if losing_trades_sum > 0 else float('inf')
            
            # Average trade duration - check if time columns exist
            if 'exit_time' in trades_df.columns and 'entry_time' in trades_df.columns:
                trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # hours
                metrics['avg_trade_duration_hours'] = trades_df['duration'].mean()
            elif 'exit_date' in trades_df.columns and 'entry_date' in trades_df.columns:
                # Try using date columns instead
                trades_df['duration'] = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.total_seconds() / 3600  # hours
                metrics['avg_trade_duration_hours'] = trades_df['duration'].mean()
            else:
                # Skip duration calculation if time columns don't exist
                metrics['avg_trade_duration_hours'] = 0
        
        # Risk metrics
        if not equity_df.empty:
            # Maximum drawdown
            equity_series = equity_df['equity']
            running_max = equity_series.cummax()
            drawdown = (equity_series / running_max - 1) * 100
            metrics['max_drawdown_pct'] = abs(drawdown.min())
            
            # Calculate Sharpe ratio (using daily returns)
            daily_returns = equity_series.pct_change().dropna()
            if len(daily_returns) > 0:
                mean_return = daily_returns.mean()
                std_return = daily_returns.std()
                risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate, converted to daily
                
                if std_return > 0:
                    metrics['sharpe_ratio'] = (mean_return - risk_free_rate) / std_return * np.sqrt(252)  # Annualized
                else:
                    metrics['sharpe_ratio'] = 0
                
                # Sortino ratio (using only negative returns for denominator)
                negative_returns = daily_returns[daily_returns < 0]
                if len(negative_returns) > 0:
                    downside_std = negative_returns.std()
                    if downside_std > 0:
                        metrics['sortino_ratio'] = (mean_return - risk_free_rate) / downside_std * np.sqrt(252)  # Annualized
                    else:
                        metrics['sortino_ratio'] = 0
                else:
                    metrics['sortino_ratio'] = float('inf')  # No negative returns
                
                # Calmar ratio (annualized return / maximum drawdown)
                if metrics['max_drawdown_pct'] > 0:
                    metrics['calmar_ratio'] = metrics['annualized_return'] / (metrics['max_drawdown_pct'] / 100)
                else:
                    metrics['calmar_ratio'] = float('inf')  # No drawdown
        
        return metrics

    def _open_position(self, symbol: str, timestamp: pd.Timestamp, price: float, direction: str):
        """
        Open a position for a given symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Entry timestamp
            price: Entry price
            direction: Position direction (long/short)
        """
        # Calculate position size
        position_size = self._calculate_position_size(symbol, price, 1)
        
        # Store original price before applying slippage
        original_price = price
        
        # Apply slippage
        price_with_slippage = self._apply_slippage(price, 1 if direction == 'long' else -1)
        
        # Calculate commission
        commission = self._apply_fee(position_size, price_with_slippage)
        
        # Update capital
        self.capital -= (position_size + commission)
        
        # Store open position
        self.open_positions[symbol] = {
            'symbol': symbol,
            'entry_time': timestamp,
            'entry_price': original_price,
            'direction': direction,
            'position_size': position_size,
            'commission': commission
        }

    def _close_position(self, symbol: str, timestamp: pd.Timestamp, price: float) -> Dict:
        """
        Close a position for a given symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Exit timestamp
            price: Exit price
            
        Returns:
            Trade details dictionary
        """
        if symbol not in self.open_positions:
            return {}
        
        position = self.open_positions[symbol]
        
        # Store original price before applying slippage
        original_exit_price = price
        
        # Apply slippage
        price_with_slippage = self._apply_slippage(price, -1)
        
        # Calculate profit/loss
        position_quantity = position['position_size'] / position['entry_price']
        
        if position['direction'] == 'long':
            pnl_pct = (original_exit_price / position['entry_price']) - 1
        else:  # short
            pnl_pct = (position['entry_price'] / original_exit_price) - 1
        
        # Calculate commission
        expected_entry_commission = position['position_size'] * self.fees['taker']
        expected_exit_commission = (position_quantity * original_exit_price) * self.fees['taker']
        total_fees = expected_entry_commission + expected_exit_commission
        
        # Calculate P&L
        pnl = position['position_size'] * pnl_pct
        
        # Update capital
        self.capital = (
            self.initial_capital - 
            position['position_size'] - 
            expected_entry_commission + 
            (position_quantity * original_exit_price) - 
            expected_exit_commission
        )
        
        # Prepare trade record
        trade = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': original_exit_price,
            'direction': position['direction'],
            'pnl': pnl - total_fees,
            'pnl_pct': pnl_pct,
            'fee': total_fees
        }
        
        # Remove position from open positions
        del self.open_positions[symbol]
        
        return trade

    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Run a single backtest.
        
        Args:
            strategy: Trading strategy to test
            data: Historical price data
            symbol: Trading symbol
            
        Returns:
            Dictionary with backtest results
        """
        # Reset backtester state
        self.reset()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize tracking variables
        equity = [self.initial_capital]
        trades = []
        
        # Simulate trading
        for i, (timestamp, row) in enumerate(data.iterrows()):
            price = row['close']
            signal = signals.iloc[i]
            
            # Check for open positions to close
            for existing_symbol in list(self.open_positions.keys()):
                trade = self._close_position(existing_symbol, timestamp, price)
                if trade:
                    trades.append(trade)
            
            # Check for trade signals
            if signal == 1 and not self.open_positions:  # Buy signal
                self._open_position(symbol, timestamp, price, 'long')
            elif signal == -1 and not self.open_positions:  # Sell signal
                self._open_position(symbol, timestamp, price, 'short')
            
            # Update equity
            portfolio_value = self.capital + sum(pos['position_size'] * price for pos in self.open_positions.values())
            equity.append(portfolio_value)
        
        # Truncate equity to match index length
        equity = equity[:len(data.index)]
        
        # Prepare results
        equity_curve = pd.Series(equity, index=data.index)
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(
            pd.DataFrame({'equity': equity_curve}), 
            pd.DataFrame(trades)
        )
        
        # Add metrics to the results
        results = {
            'equity_curve': equity_curve,
            'trades': trades,
            'signals': signals,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown_pct'],
            'win_rate': metrics['win_rate']
        }
        
        return results

    def run_multiple_backtests(self, strategy: BaseStrategy, data: pd.DataFrame, 
                                symbol: str, param_variations: Dict) -> List[Dict]:
        """
        Run multiple backtests with parameter variations.
        
        Args:
            strategy: Trading strategy to test
            data: Historical price data
            symbol: Trading symbol
            param_variations: Dictionary of parameter variations to test
            
        Returns:
            List of backtest results for each parameter combination
        """
        results = []
        
        # Generate all parameter combinations
        stop_loss_values = param_variations.get('stop_loss', [self.stop_loss])
        take_profit_values = param_variations.get('take_profit', [self.take_profit])
        
        for stop_loss in stop_loss_values:
            for take_profit in take_profit_values:
                # Temporarily modify backtester parameters
                original_stop_loss = self.stop_loss
                original_take_profit = self.take_profit
                
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                
                # Run backtest
                backtest_result = self.run_backtest(strategy, data, symbol)
                
                # Add parameter information to results
                result_with_params = {
                    'parameters': {
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    },
                    'metrics': {
                        'total_return': backtest_result['total_return'],
                        'sharpe_ratio': backtest_result['sharpe_ratio'],
                        'max_drawdown': backtest_result['max_drawdown'],
                        'win_rate': backtest_result['win_rate']
                    }
                }
                
                results.append(result_with_params)
                
                # Restore original parameters
                self.stop_loss = original_stop_loss
                self.take_profit = original_take_profit
        
        return results
    
    def plot_equity_curve(self, equity_df: pd.DataFrame, output_path: str):
        """
        Plot equity curve from backtest results.
        
        Args:
            equity_df: DataFrame with equity curve
            output_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(equity_df.index, equity_df['equity'], label='Total Equity')
        plt.plot(equity_df.index, equity_df['capital'], label='Cash', linestyle='--')
        
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Format date on x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.savefig(output_path)
        plt.close()
    
    def plot_drawdown(self, equity_df: pd.DataFrame, output_path: str):
        """
        Plot drawdown from backtest results.
        
        Args:
            equity_df: DataFrame with equity curve
            output_path: Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdown
        equity_series = equity_df['equity']
        running_max = equity_series.cummax()
        drawdown = (equity_series / running_max - 1) * 100
        
        plt.plot(equity_df.index, drawdown)
        
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Format date on x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # Fill area
        plt.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
        
        plt.savefig(output_path)
        plt.close()
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame, output_path: str):
        """
        Plot trade profit/loss distribution.
        
        Args:
            trades_df: DataFrame with trade information
            output_path: Path to save the plot
        """
        if trades_df.empty:
            logger.warning("No trades to plot distribution")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of trade P&L
        plt.hist(trades_df['pnl'], bins=20, alpha=0.7)
        
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Trade P&L Distribution')
        plt.xlabel('Profit/Loss ($)')
        plt.ylabel('Number of Trades')
        plt.grid(True)
        
        plt.savefig(output_path)
        plt.close()
    
    def plot_monthly_returns(self, equity_df: pd.DataFrame, output_path: str):
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_df: DataFrame with equity curve
            output_path: Path to save the plot
        """
        if equity_df.empty:
            logger.warning("No equity data to plot monthly returns")
            return
        
        # Calculate daily returns
        daily_returns = equity_df['equity'].pct_change().dropna()
        
        # Convert to monthly returns
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        monthly_pivot = monthly_pivot.pivot('Year', 'Month', 'Return')
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        
        # Define colormap: red for negative, green for positive
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        
        # Plot heatmap with custom colormap
        plt.pcolormesh(monthly_pivot.columns, monthly_pivot.index, monthly_pivot.values, 
                      cmap=cmap, vmin=-0.1, vmax=0.1)
        
        plt.colorbar(label='Monthly Return')
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        # Set x-tick labels to month names
        plt.xticks(np.arange(1, 13) + 0.5, 
                  ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Add text annotations with exact values
        for i in range(len(monthly_pivot.index)):
            for j in range(len(monthly_pivot.columns)):
                try:
                    value = monthly_pivot.values[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if abs(value) > 0.05 else 'black'
                        plt.text(j + 0.5, i + 0.5, f'{value:.1%}', 
                                 ha='center', va='center', color=text_color)
                except IndexError:
                    pass
        
        plt.savefig(output_path)
        plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to processed data file')
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy name to backtest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='./backtest_results',
                        help='Directory to save backtest results')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    
    return parser.parse_args()

def main():
    """Main function to run the backtester."""
    args = parse_args()
    
    # Setup logging
    log_file = f"backtest_{args.experiment}.log" if args.experiment else "backtest.log"
    setup_logger(log_file)
    
    # Create output directory
    output_dir = args.output_dir
    if args.experiment:
        output_dir = os.path.join(output_dir, args.experiment)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load data
    try:
        data = pd.read_pickle(args.data_path)
        if isinstance(data, dict) and 'clean_data' in data:
            # Extract test data if it's in the dictionary format from data_processor
            backtest_data = data['test_data']
        else:
            backtest_data = data
        
        logger.info(f"Loaded data from {args.data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create backtester
    backtester = Backtester(args.config)
    
    # Create strategy (this would typically be imported from strategies module)
    # For now, we just create a simple moving average crossover strategy
    from strategies.strategy_implementations import (
        MovingAverageCrossover, RSIThreshold, BollingerBreakout, 
        MACDStrategy, SupportResistance, CombinedStrategy
    )
    
    if args.strategy == 'MovingAverageCrossover':
        # Find strategy params in config
        strategy_params = next((s['params'] for s in config['strategies'] 
                              if s['name'] == 'MovingAverageCrossover'), {})
        strategy = MovingAverageCrossover('MA_Crossover', strategy_params)
    
    elif args.strategy == 'RSIThreshold':
        strategy_params = next((s['params'] for s in config['strategies'] 
                              if s['name'] == 'RSIThreshold'), {})
        strategy = RSIThreshold('RSI_Threshold', strategy_params)
    
    elif args.strategy == 'BollingerBreakout':
        strategy_params = next((s['params'] for s in config['strategies'] 
                              if s['name'] == 'BollingerBreakout'), {})
        strategy = BollingerBreakout('Bollinger_Breakout', strategy_params)
    
    elif args.strategy == 'MACDStrategy':
        strategy_params = next((s['params'] for s in config['strategies'] 
                              if s['name'] == 'MACDStrategy'), {})
        strategy = MACDStrategy('MACD_Strategy', strategy_params)
    
    elif args.strategy == 'SupportResistance':
        strategy_params = next((s['params'] for s in config['strategies'] 
                              if s['name'] == 'SupportResistance'), {})
        strategy = SupportResistance('Support_Resistance', strategy_params)
    
    elif args.strategy == 'Combined':
        # Create each sub-strategy
        ma_params = next((s['params'] for s in config['strategies'] 
                        if s['name'] == 'MovingAverageCrossover'), {})
        rsi_params = next((s['params'] for s in config['strategies'] 
                         if s['name'] == 'RSIThreshold'), {})
        bb_params = next((s['params'] for s in config['strategies'] 
                        if s['name'] == 'BollingerBreakout'), {})
        
        ma_strategy = MovingAverageCrossover('MA_Crossover', ma_params)
        rsi_strategy = RSIThreshold('RSI_Threshold', rsi_params)
        bb_strategy = BollingerBreakout('Bollinger_Breakout', bb_params)
        
        # Create combined strategy
        strategy_params = {
            'strategies': [ma_strategy, rsi_strategy, bb_strategy],
            'weights': [0.4, 0.3, 0.3]  # Example weights
        }
        strategy = CombinedStrategy('Combined_Strategy', strategy_params)
    
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    # Run backtest
    results = backtester.backtest(
        backtest_data, 
        strategy, 
        args.start_date, 
        args.end_date
    )
    
    # Plot results
    backtester.plot_equity_curve(
        results['equity_curve'], 
        os.path.join(output_dir, f"{args.strategy}_equity_curve.png")
    )
    
    backtester.plot_drawdown(
        results['equity_curve'], 
        os.path.join(output_dir, f"{args.strategy}_drawdown.png")
    )
    
    if not results['trades'].empty:
        backtester.plot_trade_distribution(
            results['trades'], 
            os.path.join(output_dir, f"{args.strategy}_trade_distribution.png")
        )
    
    backtester.plot_monthly_returns(
        results['equity_curve'], 
        os.path.join(output_dir, f"{args.strategy}_monthly_returns.png")
    )
    
    # Save results
    results['equity_curve'].to_csv(os.path.join(output_dir, f"{args.strategy}_equity_curve.csv"))
    results['trades'].to_csv(os.path.join(output_dir, f"{args.strategy}_trades.csv"))
    
    # Print metrics
    logger.info(f"Backtest results for {args.strategy}:")
    for key, value in results['metrics'].items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()
