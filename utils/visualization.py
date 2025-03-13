#!/usr/bin/env python
"""
Visualization utilities for the trading system.

This module provides functions for creating various types of visualizations
of market data, trading strategies, and backtest results.
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import os
import sys

# Import from the main visualization module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from visualization.visualization import (
        Visualizer,
        plot_trading_performance,
        plot_indicators,
        plot_training_metrics,
        plot_portfolio_performance,
        plot_distribution,
        plot_monte_carlo_simulations
    )
except ImportError:
    # Define placeholder functions if the main visualization module is not available
    # This ensures backward compatibility
    def plot_trading_performance(*args, **kwargs):
        """Placeholder for plot_trading_performance from visualization.visualization module."""
        logging.warning("Main visualization module not available. Using placeholder.")
        return None

    def plot_indicators(*args, **kwargs):
        """Placeholder for plot_indicators from visualization.visualization module."""
        logging.warning("Main visualization module not available. Using placeholder.")
        return None

    def plot_training_metrics(*args, **kwargs):
        """Placeholder for plot_training_metrics from visualization.visualization module."""
        logging.warning("Main visualization module not available. Using placeholder.")
        return None

    def plot_portfolio_performance(*args, **kwargs):
        """Placeholder for plot_portfolio_performance from visualization.visualization module."""
        logging.warning("Main visualization module not available. Using placeholder.")
        return None

    def plot_distribution(*args, **kwargs):
        """Placeholder for plot_distribution from visualization.visualization module."""
        logging.warning("Main visualization module not available. Using placeholder.")
        return None

    def plot_monte_carlo_simulations(*args, **kwargs):
        """Placeholder for plot_monte_carlo_simulations from visualization.visualization module."""
        logging.warning("Main visualization module not available. Using placeholder.")
        return None

    class Visualizer:
        """Placeholder for Visualizer class from visualization.visualization module."""
        def __init__(self, *args, **kwargs):
            logging.warning("Main visualization module not available. Using placeholder.")

# Configure logging
logger = logging.getLogger(__name__)

# Set Seaborn style for all plots
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


def plot_price_series(data: pd.DataFrame, title: str = "Price Series", 
                      save_path: Optional[str] = None) -> None:
    """
    Plot basic price series data.
    
    Args:
        data: DataFrame with OHLCV data
        title: Plot title
        save_path: Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(14, 8))
    
    # Check for required columns
    if 'close' not in data.columns:
        logger.error("Close price data required for plotting")
        return
    
    # Plot close price
    plt.plot(data.index, data['close'], label='Close', linewidth=2)
    
    # Add volume as a bar chart at the bottom
    if 'volume' in data.columns:
        ax2 = plt.gca().twinx()
        ax2.bar(data.index, data['volume'], alpha=0.3, color='gray', label='Volume')
        ax2.set_ylabel('Volume')
        
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Price series plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_strategy_signals(data: pd.DataFrame, signals: pd.Series, 
                          title: str = "Strategy Signals", 
                          save_path: Optional[str] = None) -> None:
    """
    Plot price data with buy/sell signals.
    
    Args:
        data: DataFrame with OHLCV data
        signals: Series with trading signals (1 for buy, -1 for sell, 0 for hold)
        title: Plot title
        save_path: Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(14, 8))
    
    # Check for required data
    if 'close' not in data.columns:
        logger.error("Close price data required for plotting")
        return
    
    # Plot close price
    plt.plot(data.index, data['close'], label='Close', linewidth=2)
    
    # Plot buy signals
    buy_signals = signals[signals == 1]
    if not buy_signals.empty:
        buy_points = data.loc[buy_signals.index, 'close']
        plt.scatter(buy_signals.index, buy_points, color='green', 
                    marker='^', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = signals[signals == -1]
    if not sell_signals.empty:
        sell_points = data.loc[sell_signals.index, 'close']
        plt.scatter(sell_signals.index, sell_points, color='red', 
                    marker='v', s=100, label='Sell Signal')
    
    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Strategy signals plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_indicators(data: pd.DataFrame, indicators: List[str],
                   title: str = "Technical Indicators", 
                   save_path: Optional[str] = None) -> None:
    """
    Plot price data with technical indicators.
    
    Args:
        data: DataFrame with OHLCV and indicator data
        indicators: List of indicator column names to plot
        title: Plot title
        save_path: Path to save the plot, if None the plot is displayed
    """
    # Make sure all indicators exist in the dataframe
    valid_indicators = [ind for ind in indicators if ind in data.columns]
    if not valid_indicators:
        logger.error("No valid indicators found for plotting")
        return
    
    n_indicators = len(valid_indicators)
    
    # Create a figure with subplots: price + each indicator
    fig, axes = plt.subplots(n_indicators + 1, 1, figsize=(14, 10), sharex=True)
    
    # Plot price in the first subplot
    if 'close' in data.columns:
        axes[0].plot(data.index, data['close'], label='Close', linewidth=2)
        axes[0].set_ylabel('Price', fontsize=12)
        axes[0].legend()
    
    # Plot each indicator in its own subplot
    for i, indicator in enumerate(valid_indicators):
        ax = axes[i + 1]
        ax.plot(data.index, data[indicator], label=indicator)
        ax.set_ylabel(indicator, fontsize=12)
        ax.legend()
    
    # Add title to overall figure
    fig.suptitle(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Indicators plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_backtest_results(equity_curve: pd.Series, trades: pd.DataFrame = None,
                          metrics: Dict = None, title: str = "Backtest Results",
                          save_path: Optional[str] = None) -> None:
    """
    Plot backtest results including equity curve and trades.
    
    Args:
        equity_curve: Series with portfolio equity values over time
        trades: DataFrame with trade information (optional)
        metrics: Dictionary of backtest metrics (optional)
        title: Plot title
        save_path: Path to save the plot, if None the plot is displayed
    """
    if trades is not None and not trades.empty:
        # Create figure with 2 rows - equity curve and drawdown
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot equity curve
        ax1.plot(equity_curve.index, equity_curve, label='Equity Curve', linewidth=2)
        
        # Add trade markers if available
        if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
            # Get profitable vs unprofitable trades
            profitable = trades[trades['profit'] > 0]
            unprofitable = trades[trades['profit'] <= 0]
            
            # Plot entry points
            if not profitable.empty:
                entries = [date for date in profitable['entry_date'] if date in equity_curve.index]
                if entries:
                    values = equity_curve.loc[entries]
                    ax1.scatter(entries, values, color='green', marker='^', s=80, label='Profitable Entry')
            
            if not unprofitable.empty:
                entries = [date for date in unprofitable['entry_date'] if date in equity_curve.index]
                if entries:
                    values = equity_curve.loc[entries]
                    ax1.scatter(entries, values, color='red', marker='^', s=80, label='Unprofitable Entry')
        
        ax1.set_title(title, fontsize=16)
        ax1.set_ylabel('Portfolio Value', fontsize=14)
        ax1.legend()
        
        # Calculate and plot drawdown
        drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)', fontsize=14)
        ax2.set_xlabel('Date', fontsize=14)
        
        # Add metrics as text if provided
        if metrics:
            metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
    else:
        # Simple equity curve plot if no trades provided
        plt.figure(figsize=(14, 8))
        plt.plot(equity_curve.index, equity_curve, label='Equity Curve', linewidth=2)
        
        # Calculate and plot drawdown
        drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        
        # Add metrics if provided
        if metrics:
            metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.gcf().text(0.05, 0.95, metrics_text, fontsize=12,
                         verticalalignment='top', bbox=props)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Portfolio Value', fontsize=14)
        plt.legend()
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Backtest results plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(results: Dict[str, pd.Series], 
                   title: str = "Strategy Comparison",
                   save_path: Optional[str] = None) -> None:
    """
    Plot comparison of multiple strategies or parameterizations.
    
    Args:
        results: Dictionary of strategy names to equity curves
        title: Plot title
        save_path: Path to save the plot, if None the plot is displayed
    """
    plt.figure(figsize=(14, 8))
    
    for name, equity_curve in results.items():
        # Normalize to starting value of 1 for fair comparison
        normalized = equity_curve / equity_curve.iloc[0]
        plt.plot(normalized.index, normalized, label=name, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Relative Performance', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Strategy comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_parameter_heatmap(results: pd.DataFrame, x_param: str, y_param: str,
                          metric: str = 'sharpe_ratio',
                          title: str = "Parameter Optimization",
                          save_path: Optional[str] = None) -> None:
    """
    Plot heatmap of performance metrics for different parameter combinations.
    
    Args:
        results: DataFrame with parameter optimization results
        x_param: Parameter name for x-axis
        y_param: Parameter name for y-axis
        metric: Performance metric to display in heatmap
        title: Plot title
        save_path: Path to save the plot, if None the plot is displayed
    """
    if x_param not in results.columns or y_param not in results.columns:
        logger.error("Parameter columns not found in results dataframe")
        return
    
    if metric not in results.columns:
        logger.error(f"Metric '{metric}' not found in results dataframe")
        return
    
    # Pivot the dataframe to create the heatmap
    pivot_table = results.pivot_table(index=y_param, columns=x_param, values=metric)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.2f')
    
    plt.title(f"{title} - {metric}", fontsize=16)
    plt.xlabel(x_param, fontsize=14)
    plt.ylabel(y_param, fontsize=14)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Parameter heatmap saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_all_plots(data: pd.DataFrame, signals: pd.Series = None,
                  indicators: List[str] = None, backtest_results: Dict = None,
                  output_dir: str = "reports/plots") -> None:
    """
    Generate and save all relevant plots for a strategy.
    
    Args:
        data: DataFrame with OHLCV and indicator data
        signals: Series with trading signals
        indicators: List of indicator column names to plot
        backtest_results: Dictionary with backtest results
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    plot_price_series(data, save_path=os.path.join(output_dir, "price_series.png"))
    
    if signals is not None:
        plot_strategy_signals(data, signals, 
                             save_path=os.path.join(output_dir, "strategy_signals.png"))
    
    if indicators:
        plot_indicators(data, indicators, 
                       save_path=os.path.join(output_dir, "technical_indicators.png"))
    
    if backtest_results:
        if 'equity_curve' in backtest_results:
            plot_backtest_results(
                backtest_results['equity_curve'],
                trades=backtest_results.get('trades'),
                metrics=backtest_results.get('metrics'),
                save_path=os.path.join(output_dir, "backtest_results.png")
            )
    
    logger.info(f"All plots saved to {output_dir}")

# Re-export functions from visualization.visualization module
__all__ = [
    'Visualizer',
    'plot_trading_performance',
    'plot_indicators',
    'plot_training_metrics',
    'plot_portfolio_performance',
    'plot_distribution',
    'plot_monte_carlo_simulations',
    'plot_price_series',
    'plot_strategy_signals',
    'plot_indicators',
    'plot_backtest_results',
    'plot_comparison',
    'plot_parameter_heatmap',
    'save_all_plots'
]
