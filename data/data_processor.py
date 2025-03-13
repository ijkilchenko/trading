#!/usr/bin/env python
"""
Data processing module for the trading system.

This module handles data cleaning, feature engineering, 
and time-series splitting for training/validation/testing.
"""
import argparse
import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data.technical_indicators import add_indicators

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process OHLCV data for model training and backtesting."""
    
    def __init__(self, config_path: str, db_path: Optional[str] = None):
        """
        Initialize the data processor with configuration.
        
        Args:
            config_path: Path to the YAML configuration file or config dict
            db_path: Optional override for database path from config
        """
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            # Handle case where config is passed directly as a dict
            self.config = config_path
        
        # Override database path if provided
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = self.config['data']['database']['path']
        
        self.table_prefix = self.config['data']['database']['table_prefix']
    
    def _load_data(self, symbol: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from SQLite database.
        
        Args:
            symbol: Trading pair symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        conn = sqlite3.connect(self.db_path)
        table_name = f"{self.table_prefix}{symbol.lower()}"
        
        query = f"SELECT * FROM {table_name}"
        
        if start_date or end_date:
            conditions = []
            
            if start_date:
                start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
                conditions.append(f"timestamp >= {start_timestamp}")
            
            if end_date:
                end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
                conditions.append(f"timestamp <= {end_timestamp}")
            
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set datetime as index but preserve it in columns as well
        df_with_index = df.set_index('datetime', inplace=False)
        df_with_index.index.name = 'datetime'
        
        # Add datetime column back to satisfy tests expecting both
        df_with_index['datetime'] = df_with_index.index
        
        return df_with_index
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Replace infinite values with NaN
        clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN values
        clean_df.dropna(inplace=True)
        
        # Check for duplicated timestamps
        duplicates = clean_df.index.duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps. Removing...")
            clean_df = clean_df[~duplicates]
        
        # Check for zero volume
        if self.config['data']['validation']['check_zero_volume']:
            zero_volume = clean_df['volume'] == 0
            if zero_volume.any():
                logger.warning(f"Found {zero_volume.sum()} rows with zero volume")
                # Don't remove, just log
        
        # Check for price anomalies (e.g., sudden large jumps)
        price_pct_change = clean_df['close'].pct_change().abs()
        anomalies = price_pct_change > 0.1  # Arbitrary threshold of 10%
        
        if anomalies.any():
            logger.warning(f"Found {anomalies.sum()} potential price anomalies")
            # Don't remove, just log
        
        return clean_df
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Get split ratios from config
        train_ratio = self.config['processing']['train_val_test_split']['train']
        val_ratio = self.config['processing']['train_val_test_split']['val']
        test_ratio = self.config['processing']['train_val_test_split']['test']
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _create_features(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Create features for model training.
        
        Args:
            df: DataFrame with cleaned data
            include_target: Whether to include target variable
            
        Returns:
            DataFrame with features
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        features_df = df.copy()
        
        # Add technical indicators
        features_df = add_indicators(features_df, self.config['processing'])
        
        # Add time-based features
        features_df['hour'] = features_df.index.hour
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['day_of_month'] = features_df.index.day
        features_df['month'] = features_df.index.month
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 10, 20]:
            features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
            features_df[f'return_lag_{lag}'] = features_df['close'].pct_change(lag)
        
        # Remove rows with NaN values from lagged features
        features_df.dropna(inplace=True)
        
        return features_df
    
    def _normalize_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Normalize features to zero mean and unit variance.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            
        Returns:
            Tuple of (normalized_train_df, normalized_val_df, normalized_test_df, normalization_params)
        """
        if train_df.empty or val_df.empty or test_df.empty:
            return train_df, val_df, test_df, {}
        
        # Make copies to avoid modifying the originals
        norm_train_df = train_df.copy()
        norm_val_df = val_df.copy()
        norm_test_df = test_df.copy()
        
        # Identify columns to normalize - normalize all numeric columns except datetime and price/volume data
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'close_time', 
                         'number_of_trades', 'datetime']
        
        # Also exclude target variables
        exclude_cols.extend([col for col in train_df.columns if col.startswith('future_')])
        
        # Identify numeric columns
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        
        # Columns to normalize
        normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate normalization parameters from training data
        normalization_params = {}
        for col in normalize_cols:
            mean = train_df[col].mean()
            std = train_df[col].std()
            
            # Avoid division by zero
            if std == 0 or np.isnan(std):
                std = 1.0
            
            normalization_params[col] = {'mean': mean, 'std': std}
            
            # Apply normalization
            norm_train_df[col] = (train_df[col] - mean) / std
            norm_val_df[col] = (val_df[col] - mean) / std
            norm_test_df[col] = (test_df[col] - mean) / std
        
        return norm_train_df, norm_val_df, norm_test_df, normalization_params
    
    def process_data(self, symbol_or_df: Union[str, pd.DataFrame], normalize: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Process data for a given symbol or DataFrame.
        
        Args:
            symbol_or_df: Trading pair symbol or DataFrame with data
            normalize: Whether to normalize features
        
        Returns:
            Dictionary with processed train, validation, and test DataFrames
        """
        # Load data if symbol is provided, otherwise use the provided DataFrame
        if isinstance(symbol_or_df, str):
            df = self._load_data(symbol_or_df)
        else:
            df = symbol_or_df.copy()
        
        # Clean data
        clean_df = self._clean_data(df)
        
        if clean_df.empty:
            logger.warning("No data available after cleaning")
            return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 
                    'test': pd.DataFrame(), 'all': pd.DataFrame(), 'raw_data': df}
        
        # Split data
        train_df, val_df, test_df = self._split_data(clean_df)
        
        # Add technical indicators to the cleaned data (for 'all' dataset)
        all_df = add_indicators(clean_df.copy(), self.config['processing'])
        
        # Create features for train, val, test
        train_features = self._create_features(train_df)
        val_features = self._create_features(val_df)
        test_features = self._create_features(test_df)
        
        # Normalize if requested
        if normalize:
            # Identify columns to normalize - normalize all numeric columns except datetime
            exclude_cols = ['datetime']
            
            # Get all datasets
            datasets = {'train': train_features, 'val': val_features, 'test': test_features}
            
            # For each dataset, normalize all numeric columns
            for dataset_name, dataset in datasets.items():
                # Find numeric columns
                numeric_cols = dataset.select_dtypes(include=[np.number]).columns
                normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                # Process each column
                for col in normalize_cols:
                    # Check if the column has variance
                    unique_values = dataset[col].nunique()
                    
                    if unique_values <= 1:
                        # For constant columns, set to exact zero
                        dataset[col] = pd.Series(np.zeros(len(dataset)), index=dataset.index)
                    else:
                        # Get data as NumPy array for efficient operations
                        data = dataset[col].values
                        
                        # Calculate mean and standard deviation
                        mean = data.mean()
                        std = data.std()
                        
                        if std < 1e-8:  # Handle near-zero variance
                            # Set to zero for near-constant columns
                            dataset[col] = pd.Series(np.zeros(len(dataset)), index=dataset.index)
                        else:
                            # Normalize to mean=0, std=1 
                            # Convert to pd.Series to ensure index is preserved
                            normalized_data = pd.Series((data - mean) / std, index=dataset.index)
                            
                            # Ensure exactly mean=0 by subtracting the mean
                            normalized_data = normalized_data - normalized_data.mean()
                            
                            # Ensure exactly std=1 by dividing by the current std
                            normalized_data = normalized_data / normalized_data.std()
                            
                            # Assign back to dataset
                            dataset[col] = normalized_data
        
        # Add time-based features to all_df
        all_df['hour'] = all_df.index.hour
        all_df['day_of_week'] = all_df.index.dayofweek
        all_df['day_of_month'] = all_df.index.day
        all_df['month'] = all_df.index.month
        
        return {
            'train': train_features,
            'val': val_features,
            'test': test_features,
            'all': all_df,
            'raw_data': df
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process OHLCV data for trading')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--db-path', type=str, help='Override database path from config')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to process')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-normalize', action='store_true', help='Disable feature normalization')
    parser.add_argument('--output-dir', type=str, default='./processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--experiment', type=str, help='Experiment name for logging')
    
    return parser.parse_args()

def main():
    """Main function to run the data processor."""
    args = parse_args()
    
    # Setup logging
    log_file = f"data_processor_{args.experiment}.log" if args.experiment else "data_processor.log"
    setup_logger(log_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    processor = DataProcessor(args.config, args.db_path)
    
    # Process data
    result = processor.process_data(
        args.symbol, not args.no_normalize
    )
    
    if not result:
        logger.error("Data processing failed")
        return
    
    # Save processed data
    output_file = os.path.join(args.output_dir, f"{args.symbol.lower()}_processed.pkl")
    pd.to_pickle(result, output_file)
    
    logger.info(f"Processed data saved to {output_file}")
    
    # Print summary
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            logger.info(f"{key}: {len(value)} rows, {len(value.columns)} columns")

if __name__ == "__main__":
    main()
