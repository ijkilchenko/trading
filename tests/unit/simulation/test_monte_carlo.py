#!/usr/bin/env python
"""
Unit tests for Monte Carlo simulation module.
"""
import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from simulation.monte_carlo import MonteCarloSimulator

class TestMonteCarloSimulator(unittest.TestCase):
    """Test suite for Monte Carlo simulator functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = cls.temp_dir.name
        
        # Create a test config
        cls.config_path = os.path.join(cls.test_dir, 'test_config.yaml')
        with open(cls.config_path, 'w') as f:
            f.write("""
simulation:
  monte_carlo:
    num_simulations: 100
    time_horizon: 30
    model: geometric_brownian_motion
    confidence_intervals: [0.5, 0.8, 0.95]
    random_seed: 42
    output_dir: {test_dir}/simulation_results
""".format(test_dir=cls.test_dir.replace('\\', '\\\\')))
        
        # Generate test price data
        cls.generate_test_data()
    
    @classmethod
    def generate_test_data(cls):
        """Generate test price data."""
        # Create a dataframe with test data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='1D')
        
        # Generate price data with trend and noise
        np.random.seed(42)  # For reproducibility
        price = 100.0
        prices = []
        
        for i in range(len(dates)):
            # Add trend and noise
            trend = 0.0005
            noise = np.random.normal(0, 0.01)
            
            # Update price
            price *= (1 + trend + noise)
            prices.append(price)
        
        # Create test dataframe
        df = pd.DataFrame({
            'open': prices * np.random.uniform(0.998, 1.002, len(prices)),
            'high': prices * np.random.uniform(1.001, 1.015, len(prices)),
            'low': prices * np.random.uniform(0.985, 0.999, len(prices)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, len(prices))
        }, index=dates)
        
        cls.test_data = df
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up for each test."""
        self.simulator = MonteCarloSimulator(self.config_path)
    
    def test_initialization(self):
        """Test simulator initialization."""
        # Check if simulator was initialized correctly
        self.assertEqual(self.simulator.num_simulations, 100)
        self.assertEqual(self.simulator.time_horizon, 30)
        self.assertEqual(self.simulator.model, 'geometric_brownian_motion')
        self.assertEqual(self.simulator.random_seed, 42)
        self.assertEqual(self.simulator.output_dir, os.path.join(self.test_dir, 'simulation_results'))
        
        # Check confidence intervals
        expected_confidence_intervals = [0.5, 0.8, 0.95]
        self.assertEqual(self.simulator.confidence_intervals, expected_confidence_intervals)
    
    def test_estimate_parameters(self):
        """Test parameter estimation from historical data."""
        # Estimate parameters
        params = self.simulator.estimate_parameters(self.test_data['close'])
        
        # Check if parameters were estimated
        self.assertIn('drift', params)
        self.assertIn('volatility', params)
        
        # Check if parameters are reasonable
        self.assertIsInstance(params['drift'], float)
        self.assertIsInstance(params['volatility'], float)
        self.assertGreaterEqual(params['volatility'], 0)
    
    def test_geometric_brownian_motion(self):
        """Test Geometric Brownian Motion simulation."""
        # Estimate parameters
        params = self.simulator.estimate_parameters(self.test_data['close'])
        
        # Simulate paths
        initial_price = self.test_data['close'].iloc[-1]
        paths = self.simulator.simulate_gbm(
            initial_price, params['drift'], params['volatility'], 
            self.simulator.time_horizon, self.simulator.num_simulations
        )
        
        # Check if paths have the right shape
        self.assertEqual(paths.shape, (self.simulator.time_horizon, self.simulator.num_simulations))
        
        # Check if all paths start at the initial price
        np.testing.assert_almost_equal(paths[0, :], initial_price, decimal=4)
        
        # Check if paths are positive
        self.assertTrue(np.all(paths > 0))
    
    def test_simulate_price_paths(self):
        """Test price path simulation for different models."""
        # Test GBM model
        self.simulator.model = 'geometric_brownian_motion'
        paths = self.simulator.simulate_price_paths(self.test_data['close'])
        
        # Check if paths have the right shape
        self.assertEqual(paths.shape, (self.simulator.time_horizon, self.simulator.num_simulations))
        
        # Test Jump Diffusion model
        self.simulator.model = 'jump_diffusion'
        paths = self.simulator.simulate_price_paths(self.test_data['close'])
        
        # Check if paths have the right shape
        self.assertEqual(paths.shape, (self.simulator.time_horizon, self.simulator.num_simulations))
        
        # Test GARCH model
        self.simulator.model = 'garch'
        paths = self.simulator.simulate_price_paths(self.test_data['close'])
        
        # Check if paths have the right shape
        self.assertEqual(paths.shape, (self.simulator.time_horizon, self.simulator.num_simulations))
        
        # Test invalid model
        self.simulator.model = 'invalid_model'
        with self.assertRaises(ValueError):
            self.simulator.simulate_price_paths(self.test_data['close'])
    
    def test_calculate_confidence_intervals(self):
        """Test calculation of confidence intervals."""
        # Simulate paths
        self.simulator.model = 'geometric_brownian_motion'
        paths = self.simulator.simulate_price_paths(self.test_data['close'])
        
        # Calculate confidence intervals
        confidence_intervals = self.simulator.calculate_confidence_intervals(
            paths, self.simulator.confidence_intervals
        )
        
        # Check if confidence intervals have the right shape
        expected_shape = (self.simulator.time_horizon, len(self.simulator.confidence_intervals) * 2)
        self.assertEqual(confidence_intervals.shape, expected_shape)
        
        # Check if lower bounds are less than upper bounds
        for i in range(len(self.simulator.confidence_intervals)):
            lower_idx = i * 2
            upper_idx = i * 2 + 1
            
            self.assertTrue(np.all(confidence_intervals[:, lower_idx] <= confidence_intervals[:, upper_idx]))
    
    def test_calculate_statistics(self):
        """Test calculation of price path statistics."""
        # Simulate paths
        self.simulator.model = 'geometric_brownian_motion'
        paths = self.simulator.simulate_price_paths(self.test_data['close'])
        
        # Calculate statistics
        stats = self.simulator.calculate_statistics(paths)
        
        # Check if stats have the right keys
        expected_keys = ['mean', 'median', 'min', 'max', 'std', 'var']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check if stats have the right shape
        self.assertEqual(stats['mean'].shape, (self.simulator.time_horizon,))
        
        # Check if statistics are reasonable
        self.assertTrue(np.all(stats['min'] <= stats['mean']))
        self.assertTrue(np.all(stats['mean'] <= stats['max']))
        self.assertTrue(np.all(stats['std'] >= 0))
    
    def test_calculate_value_at_risk(self):
        """Test calculation of Value at Risk (VaR)."""
        # Simulate paths
        self.simulator.model = 'geometric_brownian_motion'
        paths = self.simulator.simulate_price_paths(self.test_data['close'])
        
        # Calculate VaR
        confidence_level = 0.95
        var = self.simulator.calculate_value_at_risk(paths, confidence_level)
        
        # Check if VaR has the right shape
        self.assertEqual(var.shape, (self.simulator.time_horizon,))
        
        # Check if VaR is reasonable
        initial_price = paths[0, 0]
        
        # For a long position, VaR should be negative (potential loss)
        for t in range(1, self.simulator.time_horizon):
            potential_loss = var[t] - initial_price
            self.assertLessEqual(potential_loss, 0)
    
    def test_calculate_expected_shortfall(self):
        """Test calculation of Expected Shortfall (ES)."""
        # Simulate paths
        self.simulator.model = 'geometric_brownian_motion'
        paths = self.simulator.simulate_price_paths(self.test_data['close'])
        
        # Calculate ES
        confidence_level = 0.95
        es = self.simulator.calculate_expected_shortfall(paths, confidence_level)
        
        # Check if ES has the right shape
        self.assertEqual(es.shape, (self.simulator.time_horizon,))
        
        # Calculate VaR for comparison
        var = self.simulator.calculate_value_at_risk(paths, confidence_level)
        
        # ES should be more extreme than VaR
        for t in range(1, self.simulator.time_horizon):
            self.assertLessEqual(es[t], var[t])
    
    def test_run_simulation(self):
        """Test running a full simulation."""
        # Run simulation
        results = self.simulator.run_simulation(self.test_data['close'])
        
        # Check if results have the expected keys
        expected_keys = ['paths', 'confidence_intervals', 'statistics', 'var', 'es']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check if paths have the right shape
        self.assertEqual(results['paths'].shape, (self.simulator.time_horizon, self.simulator.num_simulations))
        
        # Check if confidence intervals have the right shape
        expected_ci_shape = (self.simulator.time_horizon, len(self.simulator.confidence_intervals) * 2)
        self.assertEqual(results['confidence_intervals'].shape, expected_ci_shape)

if __name__ == '__main__':
    unittest.main()
