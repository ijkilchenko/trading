import pandas as pd
import numpy as np
import sqlite3
import os
import yaml
import tempfile
from datetime import datetime, timedelta
from data.data_processor import DataProcessor

print("Creating test setup...")

# Create a test database similar to what the test is using
temp_dir = tempfile.TemporaryDirectory()
test_dir = temp_dir.name
db_path = os.path.join(test_dir, 'test_market_data.db')

# Create test configuration
config_path = os.path.join(test_dir, 'test_config.yaml')
config = {
    'data': {
        'database': {
            'path': db_path,
            'table_prefix': 'test_'
        },
        'validation': {
            'check_zero_volume': True
        }
    },
    'processing': {
        'indicators': [
            {'name': 'sma', 'params': [{'window': 10}, {'window': 20}, {'window': 50}]},
            {'name': 'ema', 'params': [{'window': 12}, {'window': 26}]},
            {'name': 'rsi', 'params': [{'window': 14}]},
            {'name': 'macd', 'params': [{'fast_period': 12, 'slow_period': 26, 'signal_period': 9}]},
            {'name': 'bollinger', 'params': [{'window': 20, 'num_std': 2}]}
        ],
        'train_val_test_split': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
    }
}

# Write config to file
with open(config_path, 'w') as f:
    yaml.safe_dump(config, f)

# Create a connection to the test database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a sample table for testing
cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_btc_usdt (
        timestamp INTEGER PRIMARY KEY,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL
    )
''')

# Generate test data with various edge cases
np.random.seed(42)
timestamps = [int((datetime(2023, 1, 1) + timedelta(minutes=i*5)).timestamp() * 1000) for i in range(1000)]
prices = [100.0]
for _ in range(1, len(timestamps)):
    # Simulate different market conditions
    trend = np.random.choice([-0.001, 0.001, 0.005, -0.005])
    volatility = np.random.uniform(0.001, 0.01)
    prices.append(prices[-1] * (1 + trend + np.random.normal(0, volatility)))

# Insert test data
for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
    open_price = price * np.random.uniform(0.99, 1.01)
    high_price = max(open_price, price * np.random.uniform(1.0, 1.05))
    low_price = min(open_price, price * np.random.uniform(0.95, 1.0))
    volume = np.random.uniform(100, 1000)
    
    cursor.execute('''
        INSERT OR REPLACE INTO test_btc_usdt 
        (timestamp, open, high, low, close, volume) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, open_price, high_price, low_price, price, volume))

conn.commit()
conn.close()

print("Test database and config created.")
print(f"Database path: {db_path}")
print(f"Config path: {config_path}")

# Initialize data processor
print("Initializing DataProcessor...")
processor = DataProcessor(config_path)

# Process data with normalization
print("Processing data with normalization...")
processed_data = processor.process_data('btc_usdt', normalize=True)

# Debug information
print("\nNumber of records in each dataset:")
for dataset_name in ['train', 'val', 'test']:
    df = processed_data[dataset_name]
    print(f"{dataset_name}: {len(df)} rows, {len(df.columns)} columns")

# Check normalization
print("\nNormalization details (focusing on failing cases):")
for dataset_name in ['train', 'val', 'test']:
    df = processed_data[dataset_name]
    
    # Check that numeric columns are normalized
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n{dataset_name} dataset:")
    
    # Count columns with std issues
    exact_std_count = 0
    close_std_count = 0
    constant_cols = 0
    
    for col in numeric_cols:
        if col != 'datetime':
            mean = df[col].mean()
            std = df[col].std()
            sample_size = len(df[col].dropna())
            
            # Exactly match the test conditions
            is_mean_close = np.isclose(mean, 0, atol=1e-5)
            is_std_close = np.isclose(std, 1, atol=1e-5)
            
            if not is_std_close:
                print(f"Column not meeting std=1 requirement: {col}")
                print(f"  mean={mean}, std={std}, samples={sample_size}")
                print(f"  is_mean_close={is_mean_close}, is_std_close={is_std_close}")
                print(f"  std diff from 1: {std - 1.0}")
                
                # Check if the column has constant value (std ≈ 0)
                if std < 1e-10:
                    constant_cols += 1
                    print(f"  Column has constant values - unique values: {df[col].nunique()}")
                    print(f"  Value counts: {df[col].value_counts().head()}")
                # Check if std is exactly 1.0
                elif std == 1.0:
                    exact_std_count += 1
                # Check if std is very close to 1.0
                elif abs(std - 1.0) < 1e-3:
                    close_std_count += 1
            else:
                exact_std_count += 1
    
    print(f"\nColumns with std exactly 1.0 or within 1e-5: {exact_std_count}/{len(numeric_cols)-1}")
    print(f"Columns with std close to 1.0 but not within tolerance: {close_std_count}")
    print(f"Columns with constant values (std ≈ 0): {constant_cols}")

# Clean up temporary directory
temp_dir.cleanup()

print("\nDone!") 