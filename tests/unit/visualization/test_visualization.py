#!/usr/bin/env python
"""
Unit tests for the visualization module.
"""
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from visualization.visualization import Visualizer

class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test environment with sample data."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a Visualizer instance
        self.visualizer = Visualizer(output_dir=self.temp_dir)
        
        # Generate sample DataFrame for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.randint(1000, 10000, len(dates)),
            'returns': np.random.normal(0.001, 0.02, len(dates)),
            'sma_20': np.random.normal(100, 5, len(dates)),  # Sample moving average
            'rsi_14': np.random.uniform(20, 80, len(dates)),  # Sample RSI
            'macd_line': np.random.normal(0, 1, len(dates)),
            'signal_line': np.random.normal(0, 1, len(dates)),
            'macd_histogram': np.random.normal(0, 1, len(dates))
        }).set_index('timestamp')

        # Sample performance metrics and training history
        self.performance_metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05
        }
        
        self.training_history = {
            'loss': [0.5, 0.4, 0.3, 0.2],
            'accuracy': [0.6, 0.7, 0.8, 0.9]
        }

    def tearDown(self):
        """Clean up temporary files and close any open plots."""
        plt.close('all')

    def _test_plot_generation(self, plot_method, *args, **kwargs):
        """
        Helper method to test plot generation with comprehensive checks.
        
        Args:
            plot_method: Method to call for plot generation
            *args: Positional arguments for the plot method
            **kwargs: Keyword arguments for the plot method
        """
        # Ensure filename is set to a path in the temp directory
        if 'filename' not in kwargs:
            kwargs['filename'] = f'test_plot_{plot_method.__name__}.png'
        
        # Generate the plot
        output_path = plot_method(*args, **kwargs)
        
        # Check if the plot was generated and saved
        self.assertIsNotNone(output_path, f"Plot generation failed for {plot_method.__name__}")
        self.assertTrue(os.path.exists(output_path), f"Plot file not created for {plot_method.__name__}")
        
        # Check file size is not zero
        self.assertGreater(os.path.getsize(output_path), 0, f"Plot file is empty for {plot_method.__name__}")
        
        # Verify the file is a valid image
        try:
            img = mpimg.imread(output_path)
            self.assertIsNotNone(img, f"Unable to read image file {output_path}")
        except Exception as e:
            self.fail(f"Error reading plot file {output_path}: {e}")

    def test_plot_training_loss(self):
        """Test training loss plot generation."""
        train_loss = [0.5, 0.4, 0.3, 0.2]
        val_loss = [0.6, 0.5, 0.4, 0.3]
        
        self._test_plot_generation(
            self.visualizer.plot_training_loss, 
            train_loss, 
            val_loss
        )

    def test_plot_metrics(self):
        """Test multiple metrics plot generation."""
        metrics = {
            'accuracy': [0.6, 0.7, 0.8, 0.9],
            'precision': [0.5, 0.6, 0.7, 0.8]
        }
        
        self._test_plot_generation(
            self.visualizer.plot_metrics, 
            metrics
        )

    def test_plot_predictions(self):
        """Test actual vs predicted values plot."""
        actual = self.sample_df['close']
        predicted = actual * 1.1  # Simple prediction
        
        self._test_plot_generation(
            self.visualizer.plot_predictions, 
            actual, 
            predicted
        )

    def test_plot_equity_curve(self):
        """Test equity curve plot generation."""
        # Create a sample equity curve DataFrame
        equity_curve = pd.DataFrame({
            'equity': np.cumsum(np.random.normal(0.001, 0.02, len(self.sample_df))),
            'capital': np.random.normal(10000, 500, len(self.sample_df))
        }, index=self.sample_df.index)
        
        self._test_plot_generation(
            self.visualizer.plot_equity_curve, 
            equity_curve
        )

    def test_plot_drawdown(self):
        """Test drawdown plot generation."""
        # Create a sample equity curve DataFrame
        equity_curve = pd.DataFrame({
            'equity': np.cumsum(np.random.normal(0.001, 0.02, len(self.sample_df)))
        }, index=self.sample_df.index)
        
        self._test_plot_generation(
            self.visualizer.plot_drawdown, 
            equity_curve
        )

    def test_plot_trade_distribution(self):
        """Test trade distribution plot generation."""
        # Create a sample trades DataFrame
        trades = pd.DataFrame({
            'pnl': np.random.normal(0, 100, 50)
        })
        
        self._test_plot_generation(
            self.visualizer.plot_trade_distribution, 
            trades
        )

    def test_plot_monthly_returns(self):
        """Test monthly returns heatmap plot generation."""
        # Create a sample equity curve DataFrame
        equity_curve = pd.DataFrame({
            'equity': np.cumsum(np.random.normal(0.001, 0.02, len(self.sample_df)))
        }, index=self.sample_df.index)
        
        self._test_plot_generation(
            self.visualizer.plot_monthly_returns, 
            equity_curve
        )

    def test_plot_technical_indicators(self):
        """Test technical indicators plot generation."""
        # Prepare indicators dictionary
        indicators = {
            'moving_averages': ['sma_20'],
            'oscillators': ['rsi_14'],
            'macd': ['macd_line', 'signal_line', 'macd_histogram']
        }
        
        self._test_plot_generation(
            self.visualizer.plot_technical_indicators, 
            self.sample_df, 
            indicators
        )

    def test_error_handling(self):
        """Test error handling for various plot scenarios."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        # Test methods with empty data
        with self.assertRaises((ValueError, RuntimeError)):
            self.visualizer.plot_equity_curve(empty_df)
        
        with self.assertRaises((ValueError, RuntimeError)):
            self.visualizer.plot_drawdown(empty_df)
        
        with self.assertRaises((ValueError, RuntimeError)):
            self.visualizer.plot_monthly_returns(empty_df)

    def test_plot_distribution(self):
        """Test distribution plot generation."""
        self._test_plot_generation(
            self.visualizer.plot_trade_distribution, 
            pd.DataFrame({'pnl': self.sample_df['returns']})
        )

if __name__ == '__main__':
    unittest.main()
