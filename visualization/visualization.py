#!/usr/bin/env python
"""
Visualization utilities for the trading system.

This module provides plotting functions for:
- Loss and metrics curves
- Trading performance
- Technical indicators
- Price data and predictions
- Monte Carlo simulations
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")

class Visualizer:
    """Visualization class for the trading system."""
    
    def __init__(self, output_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Base directory for saving plots
            experiment_name: Optional experiment name for subfolder
        """
        self.base_output_dir = output_dir
        
        # Create experiment-specific output directory if experiment name is provided
        if experiment_name:
            self.output_dir = os.path.join(output_dir, experiment_name)
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _save_figure(self, fig, filename: str):
        """
        Save a figure to the output directory.
        
        Args:
            fig: Matplotlib figure object
            filename: Name of the file to save
        """
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved figure to {filepath}")
        
        return filepath
    
    def plot_training_loss(self, train_loss: List[float], val_loss: List[float], 
                         epochs: Optional[List[int]] = None, title: str = "Training Loss Curve",
                         filename: str = "training_loss.png"):
        """
        Plot training and validation loss curves.
        
        Args:
            train_loss: List of training loss values
            val_loss: List of validation loss values
            epochs: Optional list of epoch numbers
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        if epochs is None:
            epochs = list(range(1, len(train_loss) + 1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, label='Training Loss')
        ax.plot(epochs, val_loss, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Set x-axis to show integer ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        return self._save_figure(fig, filename)
    
    def plot_metrics(self, metrics: Dict[str, List[float]], 
                    epochs: Optional[List[int]] = None, 
                    title: str = "Training Metrics", 
                    filename: str = "training_metrics.png"):
        """
        Plot multiple training metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            epochs: Optional list of epoch numbers
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        if epochs is None:
            # Use the length of the first metric
            metric_name = list(metrics.keys())[0]
            epochs = list(range(1, len(metrics[metric_name]) + 1))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name, metric_values in metrics.items():
            ax.plot(epochs, metric_values, label=metric_name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Set x-axis to show integer ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        return self._save_figure(fig, filename)
    
    def plot_predictions(self, actual: pd.Series, predicted: pd.Series, 
                        title: str = "Actual vs Predicted Values", 
                        filename: str = "predictions.png"):
        """
        Plot actual vs predicted values.
        
        Args:
            actual: Series of actual values
            predicted: Series of predicted values
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(actual.index, actual, label='Actual')
        ax.plot(predicted.index, predicted, label='Predicted')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Format date on x-axis
        if isinstance(actual.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        
        return self._save_figure(fig, filename)
    
    def plot_equity_curve(self, equity_curve: pd.DataFrame, 
                         title: str = "Equity Curve", 
                         filename: str = "equity_curve.png"):
        """
        Plot equity curve from backtest results.
        
        Args:
            equity_curve: DataFrame with equity curve
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        
        Raises:
            ValueError: If the DataFrame is empty or lacks required columns
        """
        # Check for empty DataFrame
        if equity_curve.empty:
            raise ValueError("Cannot plot equity curve with an empty DataFrame")
        
        # Check for required 'equity' column
        if 'equity' not in equity_curve.columns:
            raise ValueError("DataFrame must contain an 'equity' column")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot total equity
        ax.plot(equity_curve.index, equity_curve['equity'], label='Total Equity')
        
        # Optional: plot capital if available
        if 'capital' in equity_curve.columns:
            ax.plot(equity_curve.index, equity_curve['capital'], label='Cash', linestyle='--')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Format date on x-axis
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            fig.autofmt_xdate()
        
        return self._save_figure(fig, filename)
    
    def plot_drawdown(self, equity_curve: pd.DataFrame, 
                     title: str = "Drawdown", 
                     filename: str = "drawdown.png"):
        """
        Plot drawdown from backtest results.
        
        Args:
            equity_curve: DataFrame with equity curve
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        
        Raises:
            ValueError: If the DataFrame is empty or lacks required columns
        """
        # Check for empty DataFrame
        if equity_curve.empty:
            raise ValueError("Cannot plot drawdown with an empty DataFrame")
        
        # Check for required 'equity' column
        if 'equity' not in equity_curve.columns:
            raise ValueError("DataFrame must contain an 'equity' column")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdown
        equity_series = equity_curve['equity']
        running_max = equity_series.cummax()
        drawdown = (equity_series / running_max - 1) * 100
        
        ax.plot(equity_curve.index, drawdown)
        ax.fill_between(equity_curve.index, drawdown, 0, alpha=0.3, color='red')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title)
        ax.grid(True)
        
        # Format date on x-axis
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            fig.autofmt_xdate()
        
        return self._save_figure(fig, filename)
    
    def plot_trade_distribution(self, trades: pd.DataFrame, 
                               title: str = "Trade P&L Distribution", 
                               filename: str = "trade_distribution.png"):
        """
        Plot trade profit/loss distribution.
        
        Args:
            trades: DataFrame with trade information
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        if trades.empty:
            logger.warning("No trades to plot distribution")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of trade P&L
        ax.hist(trades['pnl'], bins=20, alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--')
        
        ax.set_xlabel('Profit/Loss ($)')
        ax.set_ylabel('Number of Trades')
        ax.set_title(title)
        ax.grid(True)
        
        return self._save_figure(fig, filename)
    
    def plot_monthly_returns(self, equity_curve: pd.DataFrame, 
                            title: str = "Monthly Returns Heatmap", 
                            filename: str = "monthly_returns.png"):
        """
        Plot monthly returns heatmap.

        Args:
            equity_curve: DataFrame with equity curve
            title: Plot title
            filename: Filename to save the plot

        Returns:
            Path to the saved figure
        """
        if equity_curve.empty:
            raise ValueError("Cannot plot monthly returns with an empty DataFrame")

        # Ensure 'equity' column exists
        if 'equity' not in equity_curve.columns:
            raise ValueError("DataFrame must contain an 'equity' column")

        # Calculate daily returns
        daily_returns = equity_curve['equity'].pct_change().dropna()

        # Convert to monthly returns
        monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })

        # Use modern pivot method
        monthly_pivot = monthly_pivot.pivot(
            index='Year', 
            columns='Month', 
            values='Return'
        )

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colormap: red for negative, green for positive
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        
        # Plot heatmap
        sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap=cmap, 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
        
        ax.set_title(title)
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set x-tick labels to month names
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        return self._save_figure(fig, filename)
    
    def plot_technical_indicators(self, data: pd.DataFrame, 
                                 indicators: Dict[str, List[str]], 
                                 title: str = "Price and Technical Indicators", 
                                 filename: str = "technical_indicators.png"):
        """
        Plot price data with technical indicators.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            indicators: Dictionary mapping indicator types to column names
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        # Create figure with subplots based on indicator types
        n_plots = 1 + len(indicators)  # Price plot + indicator plots
        fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
        
        if n_plots == 1:
            axs = [axs]  # Make axs a list if there's only one subplot
        
        # Plot price data in the first subplot
        axs[0].plot(data.index, data['close'], label='Close Price')
        
        # If we have moving averages, plot them on the price chart
        if 'moving_averages' in indicators:
            for ma in indicators['moving_averages']:
                if ma in data.columns:
                    axs[0].plot(data.index, data[ma], label=ma)
        
        # If we have Bollinger Bands, plot them on the price chart
        if 'bollinger_bands' in indicators:
            for bb in indicators['bollinger_bands']:
                if bb in data.columns:
                    axs[0].plot(data.index, data[bb], label=bb)
        
        axs[0].set_ylabel('Price')
        axs[0].set_title(title)
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot other indicators in separate subplots
        plot_idx = 1
        
        # Plot oscillators (RSI, Stochastic, etc.)
        if 'oscillators' in indicators:
            for osc in indicators['oscillators']:
                if osc in data.columns:
                    axs[plot_idx].plot(data.index, data[osc], label=osc)
                    
                    # Add overbought/oversold lines for RSI
                    if 'rsi' in osc.lower():
                        axs[plot_idx].axhline(y=70, color='r', linestyle='--')
                        axs[plot_idx].axhline(y=30, color='g', linestyle='--')
                    
                    axs[plot_idx].set_ylabel(osc)
                    axs[plot_idx].legend()
                    axs[plot_idx].grid(True)
                    plot_idx += 1
        
        # Plot MACD
        if 'macd' in indicators:
            macd_items = indicators['macd']
            if all(item in data.columns for item in macd_items):
                macd_line = data[macd_items[0]]
                signal_line = data[macd_items[1]]
                histogram = data[macd_items[2]]
                
                axs[plot_idx].plot(data.index, macd_line, label='MACD Line')
                axs[plot_idx].plot(data.index, signal_line, label='Signal Line')
                axs[plot_idx].bar(data.index, histogram, label='Histogram', alpha=0.3)
                axs[plot_idx].axhline(y=0, color='k', linestyle='-')
                
                axs[plot_idx].set_ylabel('MACD')
                axs[plot_idx].legend()
                axs[plot_idx].grid(True)
                plot_idx += 1
        
        # Format date on x-axis
        if isinstance(data.index, pd.DatetimeIndex):
            for ax in axs:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return self._save_figure(fig, filename)
    
    def plot_signals(self, data: pd.DataFrame, signals: pd.Series, 
                    title: str = "Trading Signals", 
                    filename: str = "trading_signals.png"):
        """
        Plot price data with buy/sell signals.
        
        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals (1 for buy, -1 for sell, 0 for hold)
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price data
        ax.plot(data.index, data['close'], label='Close Price')
        
        # Plot buy signals
        buy_signals = signals[signals == 1]
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'], 
                      color='green', label='Buy Signal', marker='^', s=100)
        
        # Plot sell signals
        sell_signals = signals[signals == -1]
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'], 
                      color='red', label='Sell Signal', marker='v', s=100)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Format date on x-axis
        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        
        return self._save_figure(fig, filename)
    
    def plot_monte_carlo_simulations(self, simulations: np.ndarray, 
                                    actual_prices: Optional[pd.Series] = None,
                                    title: str = "Monte Carlo Price Simulations", 
                                    filename: str = "monte_carlo_simulations.png"):
        """
        Plot Monte Carlo price simulations.
        
        Args:
            simulations: 2D array of simulated price paths (n_simulations x n_timesteps)
            actual_prices: Optional Series of actual prices for comparison
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each simulation with low alpha
        timesteps = np.arange(simulations.shape[1])
        for i in range(simulations.shape[0]):
            ax.plot(timesteps, simulations[i], color='blue', alpha=0.1)
        
        # Plot mean simulation
        mean_sim = np.mean(simulations, axis=0)
        ax.plot(timesteps, mean_sim, color='blue', label='Mean Simulation')
        
        # Plot 95% confidence interval
        percentile_5 = np.percentile(simulations, 5, axis=0)
        percentile_95 = np.percentile(simulations, 95, axis=0)
        ax.fill_between(timesteps, percentile_5, percentile_95, color='blue', alpha=0.2, label='95% Confidence Interval')
        
        # Plot actual prices if provided
        if actual_prices is not None:
            ax.plot(timesteps[:len(actual_prices)], actual_prices.values, color='red', label='Actual Prices')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return self._save_figure(fig, filename)
    
    def plot_performance_comparison(self, equity_curves: Dict[str, pd.Series], 
                                   title: str = "Strategy Performance Comparison", 
                                   filename: str = "performance_comparison.png"):
        """
        Plot performance comparison of multiple strategies.
        
        Args:
            equity_curves: Dictionary mapping strategy names to equity curves
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for strategy_name, equity_curve in equity_curves.items():
            ax.plot(equity_curve.index, equity_curve, label=strategy_name)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Format date on x-axis
        if isinstance(list(equity_curves.values())[0].index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
        
        return self._save_figure(fig, filename)
    
    def plot_correlation_matrix(self, returns: pd.DataFrame, 
                               title: str = "Strategy Returns Correlation Matrix", 
                               filename: str = "correlation_matrix.png"):
        """
        Plot correlation matrix of strategy returns.
        
        Args:
            returns: DataFrame with strategy returns (columns) over time (rows)
            title: Plot title
            filename: Filename to save the plot
            
        Returns:
            Path to the saved figure
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5, ax=ax)
        
        ax.set_title(title)
        
        return self._save_figure(fig, filename)


def plot_trading_performance(equity_curve: pd.Series, 
                            benchmark: Optional[pd.Series] = None,
                            trades: Optional[pd.DataFrame] = None,
                            metrics: Optional[Dict] = None,
                            title: str = "Trading Performance", 
                            output_path: Optional[str] = None) -> Optional[str]:
    """
    Plot trading performance metrics including equity curve and benchmark comparison.
    
    Args:
        equity_curve: Series of portfolio values over time
        benchmark: Optional benchmark series for comparison
        trades: Optional DataFrame with trade information
        metrics: Optional dictionary of performance metrics
        title: Plot title
        output_path: Path to save the output, if None returns the plot
        
    Returns:
        Path to saved file if output_path is provided, None otherwise
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    axes[0].plot(equity_curve.index, equity_curve, 'b-', label='Strategy', linewidth=2)
    
    if benchmark is not None:
        # Normalize benchmark to start at same value as strategy
        benchmark_norm = benchmark * (equity_curve.iloc[0] / benchmark.iloc[0])
        axes[0].plot(benchmark.index, benchmark_norm, 'r--', label='Benchmark', linewidth=1.5)
    
    # Plot trades if available
    if trades is not None and not trades.empty:
        for _, trade in trades.iterrows():
            if 'entry_date' in trade and 'exit_date' in trade and 'profit' in trade:
                if trade['profit'] > 0:
                    color = 'green'
                else:
                    color = 'red'
                    
                # Plot entry and exit
                if trade['entry_date'] in equity_curve.index:
                    entry_value = equity_curve.loc[trade['entry_date']]
                    axes[0].scatter(trade['entry_date'], entry_value, color=color, marker='^', s=100)
                
                if trade['exit_date'] in equity_curve.index:
                    exit_value = equity_curve.loc[trade['exit_date']]
                    axes[0].scatter(trade['exit_date'], exit_value, color=color, marker='v', s=100)
    
    axes[0].set_title(title, fontsize=16)
    axes[0].set_ylabel('Portfolio Value', fontsize=12)
    axes[0].legend()
    
    # Plot drawdown in bottom subplot
    drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
    axes[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    
    # Add metrics as text box if provided
    if metrics:
        metrics_text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return None


def plot_indicators(data: pd.DataFrame,
                  indicators: Dict[str, List[str]],
                  title: str = "Technical Indicators",
                  output_path: Optional[str] = None) -> Optional[str]:
    """
    Plot price data with technical indicators.
    
    Args:
        data: DataFrame with OHLCV and indicator data
        indicators: Dictionary mapping subplot names to lists of indicator columns
                  e.g. {"Price": ["close", "sma_20"], "Volume": ["volume"], "RSI": ["rsi_14"]}
        title: Plot title
        output_path: Path to save the output, if None returns the plot
        
    Returns:
        Path to saved file if output_path is provided, None otherwise
    """
    # Count the number of subplots needed
    n_plots = len(indicators)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)
    
    # Convert to list if only one subplot
    if n_plots == 1:
        axes = [axes]
    
    # Plot each group of indicators
    for i, (plot_name, columns) in enumerate(indicators.items()):
        ax = axes[i]
        
        for col in columns:
            if col in data.columns:
                ax.plot(data.index, data[col], label=col)
            else:
                logger.warning(f"Column {col} not found in data")
        
        ax.set_ylabel(plot_name)
        ax.legend(loc='upper left')
        
        # If this is the first subplot, add the title
        if i == 0:
            ax.set_title(title)
    
    # Format the date axis
    if not data.empty and isinstance(data.index, pd.DatetimeIndex):
        axes[-1].set_xlabel('Date')
        fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return None


def plot_training_metrics(metrics: Dict[str, List[float]],
                        epochs: Optional[List[int]] = None,
                        title: str = "Training Metrics",
                        output_path: Optional[str] = None) -> Optional[str]:
    """
    Plot training metrics over epochs.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        epochs: Optional list of epoch numbers
        title: Plot title
        output_path: Path to save the output, if None returns the plot
        
    Returns:
        Path to saved file if output_path is provided, None otherwise
    """
    # If epochs not provided, generate sequence
    max_len = max(len(values) for values in metrics.values())
    if epochs is None:
        epochs = list(range(1, max_len + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each metric
    for metric_name, values in metrics.items():
        # Ensure values list is not longer than epochs list
        values = values[:len(epochs)]
        ax.plot(epochs, values, label=metric_name)
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    
    # Set x-axis to show integer ticks for epochs
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return None


def plot_portfolio_performance(portfolio_value: pd.Series,
                             trades: Optional[pd.DataFrame] = None,
                             metrics: Optional[Dict] = None,
                             title: str = "Portfolio Performance",
                             output_path: Optional[str] = None) -> Optional[str]:
    """
    Plot portfolio performance including equity curve, trade markers and metrics.
    
    Args:
        portfolio_value: Series of portfolio values over time
        trades: Optional DataFrame with trade information
        metrics: Optional dictionary of performance metrics
        title: Plot title
        output_path: Path to save the output, if None returns the plot
        
    Returns:
        Path to saved file if output_path is provided, None otherwise
    """
    # This is a simplified version of plot_trading_performance
    # Redirecting to that function for consistency
    return plot_trading_performance(
        equity_curve=portfolio_value,
        trades=trades,
        metrics=metrics,
        title=title,
        output_path=output_path
    )


def plot_distribution(data: pd.Series,
                    title: str = "Distribution",
                    kde: bool = True,
                    output_path: Optional[str] = None) -> Optional[str]:
    """
    Plot distribution of a series using histogram and KDE.
    
    Args:
        data: Series of values to plot
        title: Plot title
        kde: Whether to include KDE curve
        output_path: Path to save the output, if None returns the plot
        
    Returns:
        Path to saved file if output_path is provided, None otherwise
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram and KDE
    sns.histplot(data, kde=kde, ax=ax)
    
    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    
    ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
    
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return None


def plot_monte_carlo_simulations(simulations: np.ndarray,
                               actual_prices: Optional[pd.Series] = None,
                               percentiles: Optional[List[float]] = None,
                               title: str = "Monte Carlo Price Simulations",
                               output_path: Optional[str] = None) -> Optional[str]:
    """
    Plot Monte Carlo simulations with percentile bands.
    
    Args:
        simulations: 2D array of simulated price paths (n_simulations x n_timesteps)
        actual_prices: Optional series of actual prices for comparison
        percentiles: Optional list of percentiles to show as bands (e.g. [5, 95])
        title: Plot title
        output_path: Path to save the output, if None returns the plot
        
    Returns:
        Path to saved file if output_path is provided, None otherwise
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Default percentiles if not provided
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    
    # Calculate time steps
    n_timesteps = simulations.shape[1]
    time_steps = np.arange(n_timesteps)
    
    # Plot a sample of simulations (max 100 for clarity)
    max_plots = min(100, simulations.shape[0])
    for i in range(max_plots):
        ax.plot(time_steps, simulations[i], color='gray', alpha=0.1)
    
    # Plot percentile bands
    percentile_values = np.percentile(simulations, percentiles, axis=0)
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    
    for i, percentile in enumerate(percentiles):
        ax.plot(time_steps, percentile_values[i], color=colors[i], 
               label=f'{percentile}th percentile', linewidth=2)
    
    # Plot actual prices if provided
    if actual_prices is not None:
        if len(actual_prices) == n_timesteps:
            ax.plot(time_steps, actual_prices, 'r-', label='Actual Prices', linewidth=3)
        else:
            logger.warning("Length of actual_prices does not match simulation timesteps")
    
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return None
