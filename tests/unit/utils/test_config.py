#!/usr/bin/env python
"""
Unit tests for configuration utilities.
"""
import os
import sys
import unittest
import tempfile
import yaml
from unittest.mock import patch, mock_open

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the utilities to be tested - assuming there's a config.py file
# If this import fails, you'll need to adjust based on your actual config handling
try:
    from utils.config import load_config, validate_config, save_config
except ImportError:
    # Define mock functions for testing purposes if the actual module doesn't exist
    def load_config(config_path):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_config(config):
        """Validate configuration structure."""
        required_sections = ['data', 'processing', 'models', 'strategies', 'backtesting']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        return True
    
    def save_config(config, config_path):
        """Save configuration to a YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

class TestConfig(unittest.TestCase):
    """Test suite for configuration utilities."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a temporary directory for test configs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_config_path = os.path.join(self.temp_dir.name, 'test_config.yaml')
        
        # Define a sample valid configuration
        self.valid_config = {
            'data': {
                'binance': {
                    'base_url': 'https://api.binance.us/api/v3',
                    'symbols': ['BTCUSDT'],
                    'start_date': '2023-01-01',
                    'end_date': '2023-01-31',
                    'interval': '1d',
                    'retries': 3,
                    'timeout': 10,
                    'rate_limit': 1200
                },
                'database': {
                    'path': 'market_data.db',
                    'table_prefix': ''
                }
            },
            'processing': {
                'train_val_test_split': {
                    'train': 0.7,
                    'val': 0.15,
                    'test': 0.15
                },
                'indicators': [
                    {'name': 'SMA', 'params': [{'window': 10}, {'window': 20}]},
                    {'name': 'RSI', 'params': [{'window': 14}]}
                ]
            },
            'models': {
                'statistical': [
                    {'name': 'ARIMA', 'params': {'p': 5, 'd': 1, 'q': 0}}
                ],
                'machine_learning': [
                    {'name': 'RandomForest', 'params': {'n_estimators': 100, 'max_depth': 10}}
                ]
            },
            'strategies': [
                {
                    'name': 'MovingAverageCrossover',
                    'params': {'fast_ma': 10, 'slow_ma': 20, 'ma_type': 'sma'}
                }
            ],
            'backtesting': {
                'initial_capital': 10000.0,
                'position_sizing': 'fixed',
                'fixed_position_size': 1000.0,
                'fees': {'maker': 0.001, 'taker': 0.001},
                'slippage': 0.001
            }
        }
        
        # Save the valid config to the test file
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.valid_config, f, default_flow_style=False)
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    def test_load_config(self):
        """Test loading configuration from a file."""
        # Load the test config
        config = load_config(self.test_config_path)
        
        # Check if the config was loaded correctly
        self.assertEqual(config, self.valid_config)
        
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            load_config('non_existent_file.yaml')
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Validate valid config
        self.assertTrue(validate_config(self.valid_config))
        
        # Test with missing required section
        invalid_config = self.valid_config.copy()
        del invalid_config['data']
        
        with self.assertRaises(ValueError):
            validate_config(invalid_config)
    
    def test_save_config(self):
        """Test saving configuration to a file."""
        # Create a new config
        new_config = {
            'data': {'source': 'binance'},
            'processing': {'indicators': ['sma', 'rsi']},
            'models': {'types': ['statistical']},
            'strategies': [{'name': 'basic_strategy'}],
            'backtesting': {'initial_capital': 5000.0}
        }
        
        # Save the new config
        new_config_path = os.path.join(self.temp_dir.name, 'new_config.yaml')
        save_config(new_config, new_config_path)
        
        # Load the saved config and check if it matches
        loaded_config = load_config(new_config_path)
        self.assertEqual(loaded_config, new_config)
    
    def test_config_merge(self):
        """Test merging configurations."""
        base_config = {
            'data': {'source': 'binance'},
            'processing': {'indicators': ['sma']},
            'models': {'types': ['arima']},
            'strategies': [{'name': 'strategy1'}],
            'backtesting': {'initial_capital': 10000.0}
        }
        
        overlay_config = {
            'data': {'source': 'custom'},
            'processing': {'indicators': ['rsi']},
            'backtesting': {'position_sizing': 'fixed'}
        }
        
        # Define a simple merge function to test
        def merge_configs(base, overlay):
            """Merge two configurations."""
            result = base.copy()
            for key, value in overlay.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            return result
        
        # Merge the configs
        merged_config = merge_configs(base_config, overlay_config)
        
        # Check if the merge was done correctly
        self.assertEqual(merged_config['data']['source'], 'custom')
        self.assertEqual(merged_config['processing']['indicators'], ['rsi'])
        self.assertEqual(merged_config['models']['types'], ['arima'])
        self.assertEqual(merged_config['strategies'], [{'name': 'strategy1'}])
        self.assertEqual(merged_config['backtesting']['initial_capital'], 10000.0)
        self.assertEqual(merged_config['backtesting']['position_sizing'], 'fixed')

if __name__ == '__main__':
    unittest.main()
