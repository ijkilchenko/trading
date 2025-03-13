#!/usr/bin/env python
"""
Trading strategy implementations for the trading system.

This module implements various trading strategies:
- Moving Average Crossover
- RSI Threshold
- Bollinger Breakout
- MACD
- Support/Resistance
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover strategy."""
    
    def __init__(self, name: str = "MovingAverageCrossover", params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
                - fast_ma: Fast moving average period
                - slow_ma: Slow moving average period
                - ma_type: Type of moving average ('sma' or 'ema')
        """
        if params is None:
            params = {'fast_ma': 20, 'slow_ma': 50, 'ma_type': 'sma'}
            
        super().__init__(name, params)
        self.fast_ma = params.get('fast_ma', 20)
        self.slow_ma = params.get('slow_ma', 50)
        self.ma_type = params.get('ma_type', 'sma')
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.Series(0, index=data.index)
        
        # Check if the required columns exist in the data
        fast_col = f"{self.ma_type}_{self.fast_ma}"
        slow_col = f"{self.ma_type}_{self.slow_ma}"
        
        if fast_col not in data.columns or slow_col not in data.columns:
            # Calculate moving averages if not present
            if self.ma_type == 'sma':
                fast_ma = data['close'].rolling(window=self.fast_ma).mean()
                slow_ma = data['close'].rolling(window=self.slow_ma).mean()
            elif self.ma_type == 'ema':
                fast_ma = data['close'].ewm(span=self.fast_ma, adjust=False).mean()
                slow_ma = data['close'].ewm(span=self.slow_ma, adjust=False).mean()
            else:
                logger.error(f"Unsupported MA type: {self.ma_type}")
                return signals
        else:
            fast_ma = data[fast_col]
            slow_ma = data[slow_col]
        
        # Calculate crossover signals
        # Buy when fast MA crosses above slow MA
        # Sell when fast MA crosses below slow MA
        
        # Previous day relationship
        prev_relationship = fast_ma.shift(1) < slow_ma.shift(1)
        
        # Current day relationship
        curr_relationship = fast_ma < slow_ma
        
        # Buy signal: Previous day fast < slow, current day fast > slow
        signals[prev_relationship & ~curr_relationship] = 1
        
        # Sell signal: Previous day fast > slow, current day fast < slow
        signals[~prev_relationship & curr_relationship] = -1
        
        return signals


class RSIThreshold(BaseStrategy):
    """RSI Threshold strategy."""
    
    def __init__(self, name: str = "RSIStrategy", params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
                - rsi_period: RSI calculation period
                - oversold: Oversold threshold
                - overbought: Overbought threshold
        """
        if params is None:
            params = {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
            
        super().__init__(name, params)
        self.rsi_period = params.get('rsi_period', 14)
        self.oversold = params.get('oversold', 30)
        self.overbought = params.get('overbought', 70)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI thresholds.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.Series(0, index=data.index)
        
        # Check if RSI column exists
        rsi_col = f"rsi_{self.rsi_period}"
        
        if rsi_col not in data.columns:
            logger.warning(f"RSI column {rsi_col} not found in data")
            return signals
        
        # Get RSI values
        rsi = data[rsi_col]
        
        # Calculate signals
        # Buy when RSI crosses below oversold threshold and then back above it
        # Sell when RSI crosses above overbought threshold and then back below it
        
        # Oversold condition: RSI < threshold
        oversold_condition = rsi < self.oversold
        
        # Overbought condition: RSI > threshold
        overbought_condition = rsi > self.overbought
        
        # Buy signal: RSI was below oversold threshold and is now crossing above it
        signals[(oversold_condition.shift(1)) & (rsi >= self.oversold)] = 1
        
        # Sell signal: RSI was above overbought threshold and is now crossing below it
        signals[(overbought_condition.shift(1)) & (rsi <= self.overbought)] = -1
        
        return signals


# Add an alias for RSIThreshold for backward compatibility
RSIStrategy = RSIThreshold


class BollingerBreakout(BaseStrategy):
    """Bollinger Bands Breakout strategy."""
    
    def __init__(self, name: str = "BollingerBandStrategy", params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
                - bb_period: Bollinger Bands calculation period
                - bb_std: Number of standard deviations
        """
        if params is None:
            params = {'bb_period': 20, 'bb_std': 2.0}
            
        super().__init__(name, params)
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands breakouts.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.Series(0, index=data.index)
        
        # Check if Bollinger Bands columns exist
        upper_band_col = f"bb_upper_{self.bb_period}_{self.bb_std}"
        middle_band_col = f"bb_middle_{self.bb_period}_{self.bb_std}"
        lower_band_col = f"bb_lower_{self.bb_period}_{self.bb_std}"
        
        required_cols = [upper_band_col, middle_band_col, lower_band_col]
        
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Bollinger Bands columns not found in data")
            return signals
        
        # Get Bollinger Bands values
        upper_band = data[upper_band_col]
        middle_band = data[middle_band_col]
        lower_band = data[lower_band_col]
        
        # Calculate band width as percentage of middle band
        band_width = (upper_band - lower_band) / middle_band
        
        # Calculate thresholds
        upper_threshold = upper_band * (1 - self.entry_threshold)
        lower_threshold = lower_band * (1 + self.entry_threshold)
        
        # Close price
        close = data['close']
        
        # Calculate signals
        # Buy when price breaks above upper band and band width is expanding
        # Sell when price breaks below lower band and band width is expanding
        
        # Band width expanding condition
        expanding = band_width > band_width.shift(1)
        
        # Buy signal: Price crosses above upper threshold with expanding band width
        signals[(close.shift(1) <= upper_threshold.shift(1)) & 
                (close > upper_threshold) & 
                expanding] = 1
        
        # Sell signal: Price crosses below lower threshold with expanding band width
        signals[(close.shift(1) >= lower_threshold.shift(1)) & 
                (close < lower_threshold) & 
                expanding] = -1
        
        return signals


# Add an alias for BollingerBreakout for backward compatibility
BollingerBandStrategy = BollingerBreakout


class MACDStrategy(BaseStrategy):
    """MACD strategy."""
    
    def __init__(self, name: str = "MACDStrategy", params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
                - fast_period: Fast EMA period
                - slow_period: Slow EMA period
                - signal_period: Signal line period
        """
        if params is None:
            params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            
        super().__init__(name, params)
        self.fast_period = params.get('fast_period', 12)
        self.slow_period = params.get('slow_period', 26)
        self.signal_period = params.get('signal_period', 9)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.Series(0, index=data.index)
        
        # Check if MACD columns exist
        macd_line_col = f"macd_line_{self.fast_period}_{self.slow_period}"
        signal_line_col = f"macd_signal_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        histogram_col = f"macd_histogram_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        
        if histogram_col in data.columns:
            # Use pre-calculated MACD histogram
            histogram = data[histogram_col]
        elif macd_line_col in data.columns and signal_line_col in data.columns:
            # Calculate histogram from line and signal
            histogram = data[macd_line_col] - data[signal_line_col]
        else:
            logger.warning(f"MACD columns not found in data")
            return signals
        
        # Calculate signals
        # Buy when histogram crosses above threshold
        # Sell when histogram crosses below negative threshold
        
        # Buy signal: Histogram crosses from below to above threshold
        signals[(histogram.shift(1) <= self.histogram_threshold) & 
                (histogram > self.histogram_threshold)] = 1
        
        # Sell signal: Histogram crosses from above to below negative threshold
        signals[(histogram.shift(1) >= -self.histogram_threshold) & 
                (histogram < -self.histogram_threshold)] = -1
        
        return signals


class SupportResistance(BaseStrategy):
    """Support and Resistance strategy."""
    
    def __init__(self, name: str = "SupportResistance", params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
                - lookback: Lookback period for finding pivots
                - threshold: Price distance threshold for clustering
                - max_levels: Maximum number of levels to consider
        """
        if params is None:
            params = {'lookback': 50, 'threshold': 0.02, 'max_levels': 5}
            
        super().__init__(name, params)
        self.lookback = params.get('lookback', 50)
        self.threshold = params.get('threshold', 0.02)
        self.max_levels = params.get('max_levels', 5)
    
    def _find_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Find local maxima for resistance levels
        resistance_levels = []
        for i in range(self.lookback, len(high) - self.lookback):
            if high.iloc[i] == high.iloc[i-self.lookback:i+self.lookback].max():
                resistance_levels.append(high.iloc[i])
        
        # Find local minima for support levels
        support_levels = []
        for i in range(self.lookback, len(low) - self.lookback):
            if low.iloc[i] == low.iloc[i-self.lookback:i+self.lookback].min():
                support_levels.append(low.iloc[i])
        
        # Cluster levels that are close to each other
        support_levels = self._cluster_levels(support_levels, close.iloc[-1])
        resistance_levels = self._cluster_levels(resistance_levels, close.iloc[-1])
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels: List[float], current_price: float) -> List[float]:
        """
        Cluster levels that are close to each other.
        
        Args:
            levels: List of price levels
            current_price: Current price
            
        Returns:
            Clustered levels
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster levels within threshold distance
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If level is close to previous level
            if (level - current_cluster[-1]) / current_price < self.threshold:
                current_cluster.append(level)
            else:
                # Add average of current cluster
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Add last cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on support and resistance levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.Series(0, index=data.index)
        
        # Loop through each day
        for i in range(self.lookback * 2, len(data)):
            # Use data up to current day to find levels
            historical_data = data.iloc[:i]
            
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_levels(historical_data)
            
            # Current and previous close prices
            current_close = data['close'].iloc[i]
            previous_close = data['close'].iloc[i-1]
            
            # Trading logic
            signal = 0
            
            # Check for support bounce
            for level in support_levels:
                # Calculate distance to level
                distance = abs(current_close - level)
                
                # If price approached support and bounced
                if (previous_close < level and 
                    (level - previous_close) / level < self.threshold and
                    current_close > previous_close):
                    signal = 1
                    break
            
            # Check for resistance bounce
            for level in resistance_levels:
                # Calculate distance to level
                distance = abs(current_close - level)
                
                # If price approached resistance and bounced down
                if (previous_close > level and 
                    (previous_close - level) / level < self.threshold and
                    current_close < previous_close):
                    signal = -1
                    break
            
            signals.iloc[i] = signal
        
        return signals


class SupportResistanceStrategy(BaseStrategy):
    """Support and Resistance strategy."""
    
    def __init__(self, name: str = "SupportResistanceStrategy", params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
                - window: Window size for support/resistance detection
                - distance_threshold: Distance to consider price near support/resistance
        """
        if params is None:
            params = {'window': 50, 'distance_threshold': 0.01}
            
        super().__init__(name, params)
        self.window = params.get('window', 50)
        self.distance_threshold = params.get('distance_threshold', 0.01)
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Identify support and resistance levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of Series with support and resistance levels
        """
        highs = data['high']
        lows = data['low']
        close = data['close']
        
        # Initialize series for support and resistance levels
        support = pd.Series(np.nan, index=data.index)
        resistance = pd.Series(np.nan, index=data.index)
        
        # Identify local minima and maxima
        for i in range(self.window, len(data) - self.window):
            # Check if this is a local minimum (potential support)
            is_min = True
            for j in range(1, self.window + 1):
                if lows.iloc[i] > lows.iloc[i-j] or lows.iloc[i] > lows.iloc[i+j]:
                    is_min = False
                    break
            
            if is_min:
                support.iloc[i] = lows.iloc[i]
            
            # Check if this is a local maximum (potential resistance)
            is_max = True
            for j in range(1, self.window + 1):
                if highs.iloc[i] < highs.iloc[i-j] or highs.iloc[i] < highs.iloc[i+j]:
                    is_max = False
                    break
            
            if is_max:
                resistance.iloc[i] = highs.iloc[i]
        
        # Forward fill to create support and resistance "zones"
        support = support.ffill()
        resistance = resistance.ffill()
        
        return support, resistance
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on support and resistance levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.Series(0, index=data.index)
        
        # Identify support and resistance levels
        support, resistance = self._identify_support_resistance(data)
        
        # Generate signals
        for i in range(1, len(data)):
            close_price = data['close'].iloc[i]
            prev_close = data['close'].iloc[i-1]
            
            # Check if price is near support or resistance
            if not pd.isna(support.iloc[i]):
                # Calculate distance to support as percentage
                support_dist = (close_price - support.iloc[i]) / support.iloc[i]
                
                # Buy signal if price is close to support and moving up
                if abs(support_dist) < self.distance_threshold and close_price > prev_close:
                    signals.iloc[i] = 1
            
            if not pd.isna(resistance.iloc[i]):
                # Calculate distance to resistance as percentage
                resistance_dist = (resistance.iloc[i] - close_price) / resistance.iloc[i]
                
                # Sell signal if price is close to resistance and moving down
                if abs(resistance_dist) < self.distance_threshold and close_price < prev_close:
                    signals.iloc[i] = -1
        
        return signals


class CombinedStrategy(BaseStrategy):
    """Combined strategy that aggregates signals from multiple strategies."""
    
    def __init__(self, name: str = "CombinedStrategy", params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
                - threshold: Signal strength threshold
        """
        if params is None:
            params = {'threshold': 0.5}
            
        super().__init__(name, params)
        self.threshold = params.get('threshold', 0.5)
        self.strategies = []
        self.weights = []
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """
        Add a strategy to the combination.
        
        Args:
            strategy: Strategy instance
            weight: Weight for the strategy
        """
        self.strategies.append(strategy)
        self.weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on weighted combination of strategies.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        if not self.strategies:
            return pd.Series(0, index=data.index)
        
        # Get signals from each strategy
        all_signals = []
        
        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            all_signals.append(signals)
        
        # Combine signals with weights
        combined_signals = pd.Series(0, index=data.index)
        
        for i, signals in enumerate(all_signals):
            weight = self.weights[i]
            combined_signals += signals * weight
        
        # Apply thresholds to determine final signal
        final_signals = pd.Series(0, index=data.index)
        
        # Buy threshold: > 0.5
        final_signals[combined_signals > 0.5] = 1
        
        # Sell threshold: < -0.5
        final_signals[combined_signals < -0.5] = -1
        
        return final_signals
