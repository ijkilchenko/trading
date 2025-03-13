#!/usr/bin/env python
"""
Technical indicators module for the trading system.

This module provides functions to calculate various technical indicators
on price data, handling edge cases and ensuring repeatability.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


def sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: Price data series
        window: Window size for SMA calculation
        
    Returns:
        Series with SMA values
    """
    return data.rolling(window=window).mean()


# Add an alias for the sma function for backward compatibility
calculate_sma = sma


def calculate_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate Simple Moving Average and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for SMA calculation
        
    Returns:
        DataFrame with added SMA column
    """
    df_copy = df.copy()
    df_copy[f'sma_{window}'] = df_copy['close'].rolling(window=window).mean()
    return df_copy


def ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: Price data series
        window: Window size for EMA calculation
        
    Returns:
        Series with EMA values
    """
    return data.ewm(span=window, adjust=False).mean()


# Add an alias for the ema function for backward compatibility
calculate_ema = ema


def calculate_ema(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate Exponential Moving Average and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for EMA calculation
        
    Returns:
        DataFrame with added EMA column
    """
    df_copy = df.copy()
    # Calculate EMA
    ema_values = df_copy['close'].ewm(span=window, adjust=False).mean()
    
    # Fill initial values with NaN for the first (window-1) periods
    ema_values.iloc[:window-1] = np.nan
    
    # Add to dataframe
    df_copy[f'ema_{window}'] = ema_values
    return df_copy


def rsi(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: Price data series
        window: Window size for RSI calculation
        
    Returns:
        Series with RSI values
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Handle division by zero
    rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss), index=data.index)
    
    # Calculate RSI
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


# Add an alias for the rsi function for backward compatibility
calculate_rsi = rsi


def calculate_rsi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculate Relative Strength Index and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for RSI calculation
        
    Returns:
        DataFrame with added RSI column
    """
    df_copy = df.copy()
    # Calculate price changes
    delta = df_copy['close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Handle division by zero
    rs = pd.Series(np.where(avg_loss == 0, 100, avg_gain / avg_loss), index=df_copy.index)
    
    # Calculate RSI
    rsi_values = 100 - (100 / (1 + rs))
    
    # Fill initial values with NaN for the first window periods
    rsi_values.iloc[:window] = np.nan
    
    df_copy[f'rsi_{window}'] = rsi_values
    
    return df_copy


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence.
    
    Args:
        data: Price data series
        fast: Fast EMA window
        slow: Slow EMA window
        signal: Signal line window
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    # Calculate fast and slow EMAs
    fast_ema = ema(data, fast)
    slow_ema = ema(data, slow)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = ema(macd_line, signal)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


# Add an alias for the macd function for backward compatibility
calculate_macd = macd


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA window
        slow: Slow EMA window
        signal: Signal line window
        
    Returns:
        DataFrame with added MACD columns
    """
    df_copy = df.copy()
    
    # Calculate fast and slow EMAs
    fast_ema = ema(df_copy['close'], fast)
    slow_ema = ema(df_copy['close'], slow)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Set initial values as NaN for the first (slow-1) periods 
    macd_line.iloc[:slow-1] = np.nan
    
    # Calculate signal line
    signal_line = ema(macd_line, signal)
    
    # Set initial values as NaN for the first (slow+signal-2) periods for signal line
    signal_line.iloc[:slow+signal-2] = np.nan
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Add to dataframe
    df_copy['macd_line'] = macd_line
    df_copy['macd_signal'] = signal_line
    df_copy['macd_histogram'] = histogram
    
    # Also add with parameters in the name for backward compatibility
    df_copy[f'macd_line_{fast}_{slow}'] = macd_line
    df_copy[f'macd_signal_{fast}_{slow}_{signal}'] = signal_line
    df_copy[f'macd_histogram_{fast}_{slow}_{signal}'] = histogram
    
    return df_copy


def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price data series
        window: Window size for moving average
        std_dev: Number of standard deviations for bands
        
    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    # Calculate middle band (SMA)
    middle_band = sma(data, window)
    
    # Calculate standard deviation
    rolling_std = data.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band


# Add an alias for the bollinger_bands function for backward compatibility
calculate_bollinger_bands = bollinger_bands


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for moving average
        std_dev: Number of standard deviations for bands
        
    Returns:
        DataFrame with added Bollinger Bands columns
    """
    df_copy = df.copy()
    
    # Calculate middle band (SMA)
    middle_band = df_copy['close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    rolling_std = df_copy['close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # Fill initial values with NaN for the first (window-1) periods
    middle_band.iloc[:window-1] = np.nan
    upper_band.iloc[:window-1] = np.nan
    lower_band.iloc[:window-1] = np.nan
    
    # Add to dataframe
    df_copy[f'bb_middle_{window}_{std_dev}'] = middle_band
    df_copy[f'bb_upper_{window}_{std_dev}'] = upper_band
    df_copy[f'bb_lower_{window}_{std_dev}'] = lower_band
    
    return df_copy


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Window size for ATR calculation
        
    Returns:
        Series with ATR values
    """
    # Calculate true range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr_values = tr.rolling(window=window).mean()
    
    return atr_values


# Add an alias for the atr function for backward compatibility
calculate_atr = atr


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for ATR calculation
        
    Returns:
        DataFrame with added ATR column
    """
    df_copy = df.copy()
    
    # Calculate true range
    prev_close = df_copy['close'].shift(1)
    tr1 = df_copy['high'] - df_copy['low']
    tr2 = (df_copy['high'] - prev_close).abs()
    tr3 = (df_copy['low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr_values = tr.ewm(span=window, adjust=False).mean()
    
    # Set first window periods to NaN
    atr_values.iloc[:window] = np.nan
    
    # Add to dataframe
    df_copy[f'atr_{window}'] = atr_values
    
    return df_copy


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                          k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_window: Window size for %K
        d_window: Window size for %D
        
    Returns:
        Tuple of (%K, %D)
    """
    # Calculate %K
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    # Handle division by zero
    denom = highest_high - lowest_low
    k = 100 * ((close - lowest_low) / denom.replace(0, np.nan))
    
    # Fill NaN values with 0 or 100 based on context
    k = k.fillna(50)  # Neutral value when there's no range
    
    # Calculate %D (SMA of %K)
    d = sma(k, d_window)
    
    return k, d


# Add an alias for the stochastic_oscillator function for backward compatibility
calculate_stochastic_oscillator = stochastic_oscillator


def calculate_stochastic_oscillator(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        k_window: Window size for %K
        d_window: Window size for %D
        
    Returns:
        DataFrame with added Stochastic Oscillator columns
    """
    df_copy = df.copy()
    
    # Calculate %K
    lowest_low = df_copy['low'].rolling(window=k_window).min()
    highest_high = df_copy['high'].rolling(window=k_window).max()
    
    # Handle division by zero
    denom = highest_high - lowest_low
    k = 100 * ((df_copy['close'] - lowest_low) / denom.replace(0, np.nan))
    
    # Fill NaN values with 0 or 100 based on context
    k = k.fillna(50)  # Neutral value when there's no range
    
    # Calculate %D (SMA of %K)
    d = sma(k, d_window)
    
    # Add to dataframe
    df_copy[f'stoch_k_{k_window}'] = k
    df_copy[f'stoch_d_{k_window}_{d_window}'] = d
    
    return df_copy


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        Series with OBV values
    """
    price_change = close.diff()
    
    # Create OBV series
    obv_values = pd.Series(0, index=close.index)
    
    # Calculate OBV
    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            obv_values.iloc[i] = obv_values.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv_values.iloc[i] = obv_values.iloc[i-1] - volume.iloc[i]
        else:
            obv_values.iloc[i] = obv_values.iloc[i-1]
    
    return obv_values


# Add an alias for the obv function for backward compatibility
calculate_obv = obv


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added OBV column
    """
    df_copy = df.copy()
    
    price_change = df_copy['close'].diff()
    
    # Create OBV series
    obv_values = pd.Series(0, index=df_copy.index)
    
    # Calculate OBV
    for i in range(1, len(df_copy)):
        if price_change.iloc[i] > 0:
            obv_values.iloc[i] = obv_values.iloc[i-1] + df_copy['volume'].iloc[i]
        elif price_change.iloc[i] < 0:
            obv_values.iloc[i] = obv_values.iloc[i-1] - df_copy['volume'].iloc[i]
        else:
            obv_values.iloc[i] = obv_values.iloc[i-1]
    
    df_copy['obv'] = obv_values
    
    return df_copy


def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                  tenkan_window: int = 9, kijun_window: int = 26,
                  senkou_span_b_window: int = 52, chikou_span_shift: int = 26) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud components.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        tenkan_window: Window size for Tenkan-sen (conversion line)
        kijun_window: Window size for Kijun-sen (base line)
        senkou_span_b_window: Window size for Senkou Span B
        chikou_span_shift: Shift for Chikou Span
        
    Returns:
        Dictionary of Ichimoku components
    """
    # Calculate Tenkan-sen (conversion line)
    tenkan_high = high.rolling(window=tenkan_window).max()
    tenkan_low = low.rolling(window=tenkan_window).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Calculate Kijun-sen (base line)
    kijun_high = high.rolling(window=kijun_window).max()
    kijun_low = low.rolling(window=kijun_window).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Calculate Senkou Span A (leading span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)
    
    # Calculate Senkou Span B (leading span B)
    senkou_high = high.rolling(window=senkou_span_b_window).max()
    senkou_low = low.rolling(window=senkou_span_b_window).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_window)
    
    # Calculate Chikou Span (lagging span)
    chikou_span = close.shift(-chikou_span_shift)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


# Add an alias for the ichimoku_cloud function for backward compatibility
calculate_ichimoku_cloud = ichimoku_cloud


def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_window: int = 9, kijun_window: int = 26,
                             senkou_span_b_window: int = 52, chikou_span_shift: int = 26) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud components and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        tenkan_window: Window size for Tenkan-sen (conversion line)
        kijun_window: Window size for Kijun-sen (base line)
        senkou_span_b_window: Window size for Senkou Span B
        chikou_span_shift: Shift for Chikou Span
        
    Returns:
        DataFrame with added Ichimoku Cloud columns
    """
    df_copy = df.copy()
    
    # Calculate Tenkan-sen (conversion line)
    tenkan_high = df_copy['high'].rolling(window=tenkan_window).max()
    tenkan_low = df_copy['low'].rolling(window=tenkan_window).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Calculate Kijun-sen (base line)
    kijun_high = df_copy['high'].rolling(window=kijun_window).max()
    kijun_low = df_copy['low'].rolling(window=kijun_window).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Calculate Senkou Span A (leading span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)
    
    # Calculate Senkou Span B (leading span B)
    senkou_high = df_copy['high'].rolling(window=senkou_span_b_window).max()
    senkou_low = df_copy['low'].rolling(window=senkou_span_b_window).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(kijun_window)
    
    # Calculate Chikou Span (lagging span)
    chikou_span = df_copy['close'].shift(-chikou_span_shift)
    
    # Add to dataframe
    df_copy['ichimoku_tenkan_sen'] = tenkan_sen
    df_copy['ichimoku_kijun_sen'] = kijun_sen
    df_copy['ichimoku_senkou_span_a'] = senkou_span_a
    df_copy['ichimoku_senkou_span_b'] = senkou_span_b
    df_copy['ichimoku_chikou_span'] = chikou_span
    
    return df_copy


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Window size for calculation
        
    Returns:
        Series with ADX values
    """
    # Calculate True Range
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    # Calculate Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    
    # Calculate +DM and -DM
    for i in range(1, len(high)):
        if high_diff.iloc[i] > 0 and high_diff.iloc[i] > abs(low_diff.iloc[i]):
            plus_dm.iloc[i] = high_diff.iloc[i]
        if low_diff.iloc[i] < 0 and abs(low_diff.iloc[i]) > high_diff.iloc[i]:
            minus_dm.iloc[i] = abs(low_diff.iloc[i])
    
    # Calculate Smoothed +DM and -DM
    smoothed_plus_dm = plus_dm.rolling(window=window).mean()
    smoothed_minus_dm = minus_dm.rolling(window=window).mean()
    
    # Calculate Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)
    
    # Calculate Directional Movement Index (DX)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # Calculate Average Directional Index (ADX)
    adx_values = dx.rolling(window=window).mean()
    
    return adx_values


# Add an alias for the adx function for backward compatibility
calculate_adx = adx


def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX) and add it to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window size for calculation
        
    Returns:
        DataFrame with added ADX column
    """
    df_copy = df.copy()
    
    # Calculate True Range
    tr1 = abs(df_copy['high'] - df_copy['low'])
    tr2 = abs(df_copy['high'] - df_copy['close'].shift(1))
    tr3 = abs(df_copy['low'] - df_copy['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    # Calculate Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    high_diff = df_copy['high'].diff()
    low_diff = df_copy['low'].diff()
    
    plus_dm = pd.Series(0.0, index=df_copy.index)
    minus_dm = pd.Series(0.0, index=df_copy.index)
    
    # Calculate +DM and -DM
    for i in range(1, len(df_copy)):
        if high_diff.iloc[i] > 0 and high_diff.iloc[i] > abs(low_diff.iloc[i]):
            plus_dm.iloc[i] = high_diff.iloc[i]
        if low_diff.iloc[i] < 0 and abs(low_diff.iloc[i]) > high_diff.iloc[i]:
            minus_dm.iloc[i] = abs(low_diff.iloc[i])
    
    # Calculate Smoothed +DM and -DM
    smoothed_plus_dm = plus_dm.rolling(window=window).mean()
    smoothed_minus_dm = minus_dm.rolling(window=window).mean()
    
    # Calculate Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)
    
    # Calculate Directional Movement Index (DX)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # Calculate Average Directional Index (ADX)
    df_copy[f'adx_{window}'] = dx.rolling(window=window).mean()
    
    return df_copy


def add_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Add technical indicators to DataFrame based on configuration.
    
    Args:
        df: DataFrame with OHLCV data
        config: Dictionary with indicator configuration
        
    Returns:
        DataFrame with added indicators
    """
    result_df = df.copy()
    
    for indicator in config['indicators']:
        name = indicator['name'].upper()
        
        for params in indicator['params']:
            if name in ['SMA', 'SIMPLE MOVING AVERAGE']:
                window = params.get('window', params.get('period', 20))
                result_df = calculate_sma(result_df, window)
            
            elif name in ['EMA', 'EXPONENTIAL MOVING AVERAGE']:
                window = params.get('window', params.get('period', 20))
                result_df = calculate_ema(result_df, window)
            
            elif name in ['RSI', 'RELATIVE STRENGTH INDEX']:
                window = params.get('window', params.get('period', 14))
                result_df = calculate_rsi(result_df, window)
            
            elif name in ['MACD', 'MOVING AVERAGE CONVERGENCE DIVERGENCE']:
                fast = params.get('fast', params.get('fast_period', 12))
                slow = params.get('slow', params.get('slow_period', 26))
                signal_window = params.get('signal', params.get('signal_period', 9))
                
                result_df = calculate_macd(result_df, fast, slow, signal_window)
            
            elif name in ['BOLLINGER', 'BBANDS', 'BOLLINGER BANDS']:
                window = params.get('window', params.get('period', 20))
                std_dev = params.get('num_std', params.get('std_dev', 2.0))
                
                result_df = calculate_bollinger_bands(result_df, window, std_dev)
            
            elif name in ['ATR', 'AVERAGE TRUE RANGE']:
                window = params.get('window', params.get('period', 14))
                result_df = calculate_atr(result_df, window)
            
            elif name in ['STOCHASTIC', 'STOCH']:
                k_window = params.get('k_window', params.get('k_period', 14))
                d_window = params.get('d_window', params.get('d_period', 3))
                
                result_df = calculate_stochastic_oscillator(result_df, k_window, d_window)
            
            elif name in ['OBV', 'ON BALANCE VOLUME']:
                result_df = calculate_obv(result_df)
                
            elif name in ['ICHIMOKU', 'ICHIMOKU CLOUD']:
                tenkan_window = params.get('tenkan_window', params.get('tenkan_period', 9))
                kijun_window = params.get('kijun_window', params.get('kijun_period', 26))
                senkou_span_b_window = params.get('senkou_span_b_window', params.get('senkou_span_b_period', 52))
                chikou_span_shift = params.get('chikou_span_shift', 26)
                
                result_df = calculate_ichimoku_cloud(result_df, tenkan_window, kijun_window, senkou_span_b_window, chikou_span_shift)
            
            elif name in ['ADX', 'AVERAGE DIRECTIONAL INDEX']:
                window = params.get('window', params.get('period', 14))
                result_df = calculate_adx(result_df, window)
    
    return result_df
