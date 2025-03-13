#!/usr/bin/env python
"""
Live trading module for the trading system.

This module executes trading strategies in real-time,
supporting both paper trading and live trading modes.
"""
import argparse
import datetime
import json
import logging
import os
import signal
import sys
import threading
import time
import yaml
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple, Union
import sqlite3
from decimal import Decimal, ROUND_DOWN

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from strategies.base_strategy import BaseStrategy
from data.data_processor import DataProcessor
from data.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class Trader:
    """Live trading implementation."""
    
    def __init__(self, config_path: str, mode: str = 'paper'):
        """
        Initialize the trader.
        
        Args:
            config_path: Path to the YAML configuration file
            mode: Trading mode ('paper' or 'live')
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.mode = mode
        self.running = False
        self.last_update_time = None
        
        # Initialize parameters from config
        self.trading_config = self.config['trading']
        self.symbols = self.trading_config['symbols']
        self.update_interval = self.trading_config['update_interval']  # in seconds
        self.db_path = self.trading_config['db_path']
        
        # Initialize trading account parameters
        if mode == 'paper':
            self.paper_config = self.trading_config['paper']
            self.balance = self.paper_config['initial_balance']
            self.positions = {symbol: 0 for symbol in self.symbols}
            self.trades = []
        else:
            # For live trading, initialize API keys
            self.api_config = self.trading_config['api']
            self.api_key = self.api_config['api_key']
            self.api_secret = self.api_config['api_secret']
        
        # Initialize trading parameters
        self.position_sizing = self.trading_config['position_sizing']
        self.fixed_position_size = self.trading_config['fixed_position_size']
        self.percentage_position_size = self.trading_config['percentage_position_size']
        self.maker_fee = self.trading_config['fees']['maker']
        self.taker_fee = self.trading_config['fees']['taker']
        
        # Initialize data processor
        self.data_processor = DataProcessor(config_path)
        
        # Initialize technical indicators
        self.ti = TechnicalIndicators()
        
        # Create database connection
        self.conn = sqlite3.connect(self.db_path)
        
        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals to gracefully shut down."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.stop()
    
    def load_strategy(self, strategy: BaseStrategy):
        """
        Load trading strategy.
        
        Args:
            strategy: Trading strategy instance
        """
        self.strategy = strategy
        logger.info(f"Loaded strategy: {strategy.name}")
    
    def _calculate_position_size(self, capital: float, price: float, symbol: str) -> float:
        """
        Calculate position size based on position sizing strategy.
        
        Args:
            capital: Available capital
            price: Current price
            symbol: Trading symbol
            
        Returns:
            Position size in units
        """
        if self.position_sizing == 'fixed':
            # Fixed dollar amount
            return self.fixed_position_size / price
        elif self.position_sizing == 'percentage':
            # Percentage of capital
            return (capital * self.percentage_position_size) / price
        else:
            # Default to fixed
            logger.warning(f"Unknown position sizing strategy: {self.position_sizing}. Using fixed.")
            return self.fixed_position_size / price
    
    def _round_decimals(self, value: float, decimals: int = 8) -> float:
        """
        Round a float to a specific number of decimal places.
        
        Args:
            value: Float value to round
            decimals: Number of decimal places
            
        Returns:
            Rounded float
        """
        return float(Decimal(str(value)).quantize(Decimal('0.' + '0' * decimals), rounding=ROUND_DOWN))
    
    def _fetch_latest_data(self, symbol: str, lookback_period: int = 200) -> pd.DataFrame:
        """
        Fetch latest market data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            lookback_period: Number of historical bars to fetch
            
        Returns:
            DataFrame with latest market data
        """
        try:
            logger.info(f"Fetching latest data for {symbol}")
            
            # For paper trading, fetch data from our database
            if self.mode == 'paper':
                query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT {lookback_period}
                """
                
                df = pd.read_sql_query(query, self.conn)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)  # Sort by timestamp in ascending order
                
                return df
            
            # For live trading, fetch data from Binance
            else:
                # Convert symbol format for Binance (e.g., 'BTC-USD' -> 'BTCUSD')
                binance_symbol = symbol.replace('-', '')
                
                # Kline endpoint
                endpoint = "https://api.binance.us/api/v3/klines"
                
                # Calculate start time (in milliseconds)
                current_time = int(time.time() * 1000)
                start_time = current_time - (lookback_period * 60 * 1000)  # 1-minute intervals
                
                # API parameters
                params = {
                    'symbol': binance_symbol,
                    'interval': '1m',
                    'startTime': start_time,
                    'limit': lookback_period
                }
                
                # Make API request
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Create DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                # Set index
                df.set_index('timestamp', inplace=True)
                
                return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            DataFrame with added indicators
        """
        try:
            logger.debug(f"Adding indicators for {symbol}")
            
            # Get indicator parameters from the strategy
            # This assumes strategy has params for the indicators it needs
            strategy_params = self.strategy.params
            
            # Add indicators based on strategy requirements
            
            # Moving Averages
            if 'fast_ma' in strategy_params and 'slow_ma' in strategy_params:
                fast_ma = strategy_params.get('fast_ma', 20)
                slow_ma = strategy_params.get('slow_ma', 50)
                ma_type = strategy_params.get('ma_type', 'sma')
                
                if ma_type == 'sma':
                    data[f'sma_{fast_ma}'] = self.ti.sma(data['close'], fast_ma)
                    data[f'sma_{slow_ma}'] = self.ti.sma(data['close'], slow_ma)
                elif ma_type == 'ema':
                    data[f'ema_{fast_ma}'] = self.ti.ema(data['close'], fast_ma)
                    data[f'ema_{slow_ma}'] = self.ti.ema(data['close'], slow_ma)
            
            # RSI
            if 'rsi_period' in strategy_params:
                rsi_period = strategy_params.get('rsi_period', 14)
                data[f'rsi_{rsi_period}'] = self.ti.rsi(data['close'], rsi_period)
            
            # Bollinger Bands
            if 'window' in strategy_params and 'std_dev' in strategy_params:
                window = strategy_params.get('window', 20)
                std_dev = strategy_params.get('std_dev', 2)
                bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(data['close'], window, std_dev)
                data[f'bb_upper_{window}_{std_dev}'] = bb_upper
                data[f'bb_middle_{window}_{std_dev}'] = bb_middle
                data[f'bb_lower_{window}_{std_dev}'] = bb_lower
            
            # MACD
            if 'fast' in strategy_params and 'slow' in strategy_params and 'signal' in strategy_params:
                fast = strategy_params.get('fast', 12)
                slow = strategy_params.get('slow', 26)
                signal_period = strategy_params.get('signal', 9)
                macd_line, signal_line, histogram = self.ti.macd(data['close'], fast, slow, signal_period)
                data[f'macd_line_{fast}_{slow}'] = macd_line
                data[f'macd_signal_{fast}_{slow}_{signal_period}'] = signal_line
                data[f'macd_histogram_{fast}_{slow}_{signal_period}'] = histogram
            
            return data
        
        except Exception as e:
            logger.error(f"Error adding indicators for {symbol}: {e}")
            return data
    
    def _execute_trade(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """
        Execute a trade either in paper or live mode.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            quantity: Quantity to trade
            price: Current price
            
        Returns:
            Trade execution details
        """
        trade_time = datetime.datetime.now()
        
        if self.mode == 'paper':
            # Paper trading logic
            value = quantity * price
            fee = value * self.taker_fee  # Assume taker fee
            
            if side == 'buy':
                # Check if enough balance
                if value + fee > self.balance:
                    logger.warning(f"Insufficient balance for buy order. Need ${value + fee:.2f}, have ${self.balance:.2f}")
                    return None
                
                # Update balance and position
                self.balance -= (value + fee)
                self.positions[symbol] += quantity
                
                trade = {
                    'timestamp': trade_time,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'value': value,
                    'fee': fee
                }
                
                self.trades.append(trade)
                logger.info(f"PAPER TRADE: BUY {quantity} {symbol} @ ${price:.2f}, Total: ${value:.2f}, Fee: ${fee:.2f}")
                
                return trade
                
            elif side == 'sell':
                # Check if enough position
                if quantity > self.positions[symbol]:
                    logger.warning(f"Insufficient position for sell order. Need {quantity} {symbol}, have {self.positions[symbol]}")
                    return None
                
                # Update balance and position
                self.balance += (value - fee)
                self.positions[symbol] -= quantity
                
                trade = {
                    'timestamp': trade_time,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'value': value,
                    'fee': fee
                }
                
                self.trades.append(trade)
                logger.info(f"PAPER TRADE: SELL {quantity} {symbol} @ ${price:.2f}, Total: ${value:.2f}, Fee: ${fee:.2f}")
                
                return trade
        else:
            # Live trading logic
            try:
                # Convert symbol format for Binance (e.g., 'BTC-USD' -> 'BTCUSD')
                binance_symbol = symbol.replace('-', '')
                
                # Create order endpoint
                endpoint = "https://api.binance.us/api/v3/order"
                
                # API parameters for market order
                params = {
                    'symbol': binance_symbol,
                    'side': side.upper(),
                    'type': 'MARKET',
                    'quantity': quantity,
                    'timestamp': int(time.time() * 1000)
                }
                
                # TODO: Implement API signature creation for authenticated requests
                # This would involve creating an HMAC SHA256 signature using the API secret
                
                # Make API request (disabled for safety in this implementation)
                # response = requests.post(endpoint, params=params, headers={'X-MBX-APIKEY': self.api_key})
                # response.raise_for_status()
                # order_details = response.json()
                
                # Instead, log the intended action
                logger.info(f"LIVE TRADE (SIMULATION): {side.upper()} {quantity} {symbol} @ market price")
                
                # Return simulated order details
                return {
                    'timestamp': trade_time,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,  # Estimated price
                    'value': quantity * price,
                    'fee': quantity * price * self.taker_fee
                }
                
            except Exception as e:
                logger.error(f"Error executing {side} order for {symbol}: {e}")
                return None
    
    def _process_symbol(self, symbol: str):
        """
        Process a trading symbol to generate and execute signals.
        
        Args:
            symbol: Trading symbol
        """
        try:
            # Fetch latest data for the symbol
            data = self._fetch_latest_data(symbol)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Add technical indicators
            data = self._add_indicators(data, symbol)
            
            # Generate signals
            signals = self.strategy.generate_signals(data)
            
            # Get the latest signal
            latest_signal = signals.iloc[-1] if not signals.empty else 0
            
            # Get current position
            current_position = self.positions[symbol] if self.mode == 'paper' else self._get_live_position(symbol)
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Execute trade if signal is non-zero
            if latest_signal == 1 and current_position <= 0:  # Buy signal
                # Calculate position size
                if current_position < 0:  # Close short position first
                    quantity = abs(current_position)
                    self._execute_trade(symbol, 'buy', quantity, current_price)
                
                # Open new long position
                capital = self.balance if self.mode == 'paper' else self._get_available_balance()
                position_size = self._calculate_position_size(capital, current_price, symbol)
                
                # Round to appropriate decimals for the symbol
                position_size = self._round_decimals(position_size, 6)  # Adjust decimals as needed for the asset
                
                if position_size > 0:
                    self._execute_trade(symbol, 'buy', position_size, current_price)
            
            elif latest_signal == -1 and current_position >= 0:  # Sell signal
                # Calculate position size
                if current_position > 0:  # Close long position first
                    quantity = current_position
                    self._execute_trade(symbol, 'sell', quantity, current_price)
                
                # Open new short position (only for paper trading)
                if self.mode == 'paper':
                    capital = self.balance
                    position_size = self._calculate_position_size(capital, current_price, symbol)
                    
                    # Round to appropriate decimals for the symbol
                    position_size = self._round_decimals(position_size, 6)
                    
                    if position_size > 0:
                        self._execute_trade(symbol, 'sell', position_size, current_price)
            
            # Log current status
            if self.mode == 'paper':
                logger.info(f"Current balance: ${self.balance:.2f}, {symbol} position: {self.positions[symbol]}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def _get_live_position(self, symbol: str) -> float:
        """
        Get current position size for a symbol in live trading.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position size
        """
        try:
            # Convert symbol format for Binance
            binance_symbol = symbol.replace('-', '')
            
            # Account information endpoint
            endpoint = "https://api.binance.us/api/v3/account"
            
            # TODO: Implement API signature creation for authenticated requests
            
            # For now, return 0 as a placeholder
            logger.info(f"Getting live position for {symbol} (simulated)")
            return 0
            
            # In a real implementation, we would parse the account balances
            # and return the actual position
            
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return 0
    
    def _get_available_balance(self) -> float:
        """
        Get available balance for trading in live mode.
        
        Returns:
            Available balance in USD
        """
        try:
            # Account information endpoint
            endpoint = "https://api.binance.us/api/v3/account"
            
            # TODO: Implement API signature creation for authenticated requests
            
            # For now, return a placeholder value
            logger.info("Getting available balance (simulated)")
            return 10000.0
            
            # In a real implementation, we would parse the account balances
            # and return the actual USD balance
            
        except Exception as e:
            logger.error(f"Error getting available balance: {e}")
            return 0
    
    def _update_status(self):
        """Update trading status and save to file."""
        if self.mode == 'paper':
            status = {
                'timestamp': datetime.datetime.now().isoformat(),
                'balance': self.balance,
                'positions': self.positions,
                'trades': len(self.trades)
            }
            
            try:
                with open('paper_trading_status.json', 'w') as f:
                    # Convert positions to serializable format
                    serializable_status = {
                        'timestamp': status['timestamp'],
                        'balance': status['balance'],
                        'positions': {k: float(v) for k, v in status['positions'].items()},
                        'trades': status['trades']
                    }
                    json.dump(serializable_status, f, indent=4)
            except Exception as e:
                logger.error(f"Error saving trading status: {e}")
    
    def _save_paper_trades(self):
        """Save paper trading history to CSV file."""
        if self.mode == 'paper' and self.trades:
            try:
                df = pd.DataFrame(self.trades)
                df.to_csv('paper_trading_history.csv', index=False)
                logger.info(f"Saved {len(self.trades)} paper trades to CSV")
            except Exception as e:
                logger.error(f"Error saving paper trades: {e}")
    
    def start(self):
        """Start the trading loop."""
        if not hasattr(self, 'strategy'):
            logger.error("No strategy loaded. Cannot start trading.")
            return
        
        self.running = True
        logger.info(f"Starting {self.mode} trading...")
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check if we should update
                if self.last_update_time is None or (current_time - self.last_update_time) >= self.update_interval:
                    logger.info(f"Running trading update at {datetime.datetime.now()}")
                    
                    # Process each symbol
                    for symbol in self.symbols:
                        self._process_symbol(symbol)
                    
                    # Update status
                    self._update_status()
                    
                    # Update last update time
                    self.last_update_time = current_time
                
                # Sleep to avoid excessive CPU usage
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user.")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.stop()
    
    def start_async(self):
        """Start the trading loop in a separate thread."""
        self.trading_thread = threading.Thread(target=self.start)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        logger.info(f"Started {self.mode} trading in background thread")
    
    def stop(self):
        """Stop the trading loop."""
        self.running = False
        logger.info("Stopping trading...")
        
        # Save paper trading history
        if self.mode == 'paper':
            self._save_paper_trades()
        
        # Close database connection
        if hasattr(self, 'conn'):
            self.conn.close()
        
        logger.info("Trading stopped")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run trading strategy')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['paper', 'live'], default='paper',
                        help='Trading mode')
    parser.add_argument('--strategy', type=str, required=True,
                        help='Strategy name to use')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level')
    
    return parser.parse_args()

def main():
    """Main function to run the trader."""
    args = parse_args()
    
    # Setup logging
    log_file = f"{args.mode}_trading.log"
    setup_logger(log_file, level=args.log_level)
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create trader
    trader = Trader(args.config, args.mode)
    
    # Create strategy (this would typically be imported from strategies module)
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
    
    # Load strategy
    trader.load_strategy(strategy)
    
    # Start trading
    trader.start()

if __name__ == "__main__":
    main()
