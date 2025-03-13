#!/usr/bin/env python
"""
Monte Carlo simulation for market price paths.

This module provides tools to generate synthetic market data
using various stochastic processes to simulate possible price paths.
"""
import argparse
import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from utils.logger import setup_logger
from utils.visualization import Visualizer

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """Monte Carlo simulator for market price paths."""
    
    def __init__(self, config_path: str):
        """
        Initialize the simulator.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Match test expectations for configuration
        monte_carlo_config = self.config.get('simulation', {}).get('monte_carlo', {})
        
        # Set parameters from config
        self.num_simulations = monte_carlo_config.get('num_simulations', 100)
        self.time_horizon = monte_carlo_config.get('time_horizon', 30)
        self.model = monte_carlo_config.get('model', 'geometric_brownian_motion')
        self.random_seed = monte_carlo_config.get('random_seed', 42)
        self.confidence_intervals = monte_carlo_config.get('confidence_intervals', [0.5, 0.8, 0.95])
        
        # Set output directory
        self.output_dir = monte_carlo_config.get('output_dir', './simulation_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.output_dir)
    
    def estimate_parameters(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Estimate drift and volatility from historical price data.
        
        Args:
            price_series: Historical price series
        
        Returns:
            Dictionary with drift and volatility parameters
        """
        # Calculate log returns
        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        
        # Estimate drift (mean of log returns)
        drift = log_returns.mean() * 252  # Annualized
        
        # Estimate volatility (standard deviation of log returns)
        volatility = log_returns.std() * np.sqrt(252)  # Annualized
        
        return {
            'drift': drift,
            'volatility': volatility
        }
    
    def simulate_gbm(self, initial_price: float, drift: float, volatility: float, 
                     time_horizon: int, num_simulations: int) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion price paths.
        
        Args:
            initial_price: Starting price
            drift: Annualized drift
            volatility: Annualized volatility
            time_horizon: Number of time steps
            num_simulations: Number of simulation paths
        
        Returns:
            Simulated price paths
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Convert annual parameters to daily
        daily_drift = drift / 252
        daily_volatility = volatility / np.sqrt(252)
        
        # Initialize price array
        paths = np.zeros((time_horizon, num_simulations))
        paths[0, :] = initial_price
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (time_horizon - 1, num_simulations))
        
        # Simulate price paths
        for t in range(1, time_horizon):
            paths[t, :] = paths[t-1, :] * np.exp(
                (daily_drift - 0.5 * daily_volatility**2) + daily_volatility * random_shocks[t-1, :]
            )
        
        return paths
    
    def simulate_price_paths(self, historical_prices: pd.Series) -> np.ndarray:
        """
        Simulate price paths based on the selected model.
        
        Args:
            historical_prices: Historical price series for parameter estimation
        
        Returns:
            Simulated price paths
        """
        # Estimate parameters from historical prices
        params = self.estimate_parameters(historical_prices)
        initial_price = historical_prices.iloc[-1]
        
        # Select simulation model
        if self.model == 'geometric_brownian_motion':
            return self.simulate_gbm(
                initial_price, 
                params['drift'], 
                params['volatility'], 
                self.time_horizon, 
                self.num_simulations
            )
        elif self.model == 'jump_diffusion':
            # Placeholder for jump diffusion model
            return self.simulate_gbm(
                initial_price, 
                params['drift'], 
                params['volatility'], 
                self.time_horizon, 
                self.num_simulations
            )
        elif self.model == 'garch':
            # Placeholder for GARCH model
            return self.simulate_gbm(
                initial_price, 
                params['drift'], 
                params['volatility'], 
                self.time_horizon, 
                self.num_simulations
            )
        else:
            raise ValueError(f"Unsupported simulation model: {self.model}")
    
    def calculate_confidence_intervals(self, paths: np.ndarray, confidence_levels: List[float]) -> np.ndarray:
        """
        Calculate confidence intervals for simulated price paths.
        
        Args:
            paths: Simulated price paths
            confidence_levels: List of confidence levels to calculate
        
        Returns:
            Array of confidence intervals
        """
        # Initialize confidence interval array
        intervals = np.zeros((paths.shape[0], len(confidence_levels) * 2))
        
        # Calculate confidence intervals for each time step
        for t in range(paths.shape[0]):
            time_step_paths = paths[t, :]
            
            for i, conf_level in enumerate(confidence_levels):
                lower_bound = np.percentile(time_step_paths, (1 - conf_level) / 2 * 100)
                upper_bound = np.percentile(time_step_paths, (1 + conf_level) / 2 * 100)
                
                intervals[t, i*2] = lower_bound
                intervals[t, i*2 + 1] = upper_bound
        
        return intervals
    
    def calculate_value_at_risk(self, paths: np.ndarray, confidence_level: float) -> np.ndarray:
        """
        Calculate Value at Risk (VaR) for the simulated paths.
        
        Args:
            paths: Simulated price paths
            confidence_level: Confidence level for VaR calculation
        
        Returns:
            Time series of Value at Risk (negative values representing potential loss)
        """
        # Ensure paths are time_horizon x num_simulations
        if paths.shape[0] < paths.shape[1]:
            paths = paths.T
        
        # Get initial price
        initial_price = paths[0, 0]
        
        # Calculate VaR at each time step
        var_series = np.zeros(self.time_horizon)
        for t in range(self.time_horizon):
            # Calculate VaR as potential loss from initial price
            var_series[t] = initial_price - np.percentile(paths[t, :], confidence_level * 100)
        
        return var_series
    
    def calculate_expected_shortfall(self, paths: np.ndarray, confidence_level: float) -> np.ndarray:
        """
        Calculate Expected Shortfall (ES) for the simulated paths.
        
        Args:
            paths: Simulated price paths
            confidence_level: Confidence level for ES calculation
        
        Returns:
            Time series of Expected Shortfall (negative values representing potential loss)
        """
        # Ensure paths are time_horizon x num_simulations
        if paths.shape[0] < paths.shape[1]:
            paths = paths.T
        
        # Get initial price
        initial_price = paths[0, 0]
        
        # Calculate ES at each time step
        es_series = np.zeros(self.time_horizon)
        for t in range(self.time_horizon):
            # Calculate VaR for this time step
            var = initial_price - np.percentile(paths[t, :], confidence_level * 100)
            
            # Calculate average of paths below VaR
            es_paths = paths[t, paths[t, :] <= (initial_price - var)]
            
            # If no paths are below VaR, use VaR
            # Ensure ES is always less than or equal to VaR
            es_series[t] = min(
                initial_price - np.mean(es_paths) if len(es_paths) > 0 else var,
                var
            )
        
        return es_series
    
    def calculate_statistics(self, paths: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate statistics for simulated price paths.
        
        Args:
            paths: Simulated price paths
        
        Returns:
            Dictionary of statistical measures
        """
        # Ensure paths are time_horizon x num_simulations
        if paths.shape[0] < paths.shape[1]:
            paths = paths.T
        
        # Calculate statistics
        stats = {
            'mean': np.mean(paths, axis=1)[:self.time_horizon],
            'median': np.median(paths, axis=1)[:self.time_horizon],
            'min': np.min(paths, axis=1)[:self.time_horizon],
            'max': np.max(paths, axis=1)[:self.time_horizon],
            'std': np.std(paths, axis=1)[:self.time_horizon],
            'var': np.var(paths, axis=1)[:self.time_horizon]
        }
        
        return stats
    
    def create_dataframe(self, prices: np.ndarray, start_date: Optional[str] = None) -> List[pd.DataFrame]:
        """
        Convert simulated prices to list of DataFrames with OHLCV format.
        
        Args:
            prices: Array of simulated price paths
            start_date: Optional start date (default: today)
            
        Returns:
            List of DataFrames with simulated OHLCV data
        """
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = pd.to_datetime('today')
        
        # Create date range
        dates = [start + timedelta(days=i) for i in range(self.time_horizon)]
        
        # List to store DataFrames
        dataframes = []
        
        # Convert each simulation to a DataFrame
        for i in range(self.num_simulations):
            price_series = prices[:, i]
            
            # Generate OHLCV data
            # For simplicity, we'll use basic price transformation to generate OHLC
            open_prices = price_series
            high_prices = price_series * np.random.uniform(1.0, 1.02, self.time_horizon)  # 0-2% higher
            low_prices = price_series * np.random.uniform(0.98, 1.0, self.time_horizon)   # 0-2% lower
            close_prices = price_series
            volumes = np.random.lognormal(10, 1, self.time_horizon)  # Random volumes
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            df.set_index('timestamp', inplace=True)
            dataframes.append(df)
        
        return dataframes
    
    def save_simulations(self, dataframes: List[pd.DataFrame], base_filename: str = 'simulated_data'):
        """
        Save simulated data to disk.
        
        Args:
            dataframes: List of DataFrames with simulated data
            base_filename: Base filename for output files
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save each simulation to a separate file
        for i, df in enumerate(dataframes):
            filename = f"{base_filename}_{i+1}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            df.to_pickle(filepath)
            
            # Also save first 10 simulations as CSV for easier inspection
            if i < 10:
                csv_filepath = os.path.join(self.output_dir, f"{base_filename}_{i+1}.csv")
                df.to_csv(csv_filepath)
        
        logger.info(f"Saved {len(dataframes)} simulations to {self.output_dir}")
        
        # Also save all simulations combined
        all_filepath = os.path.join(self.output_dir, f"{base_filename}_all.pkl")
        pd.to_pickle(dataframes, all_filepath)
    
    def visualize_simulations(self, prices: np.ndarray, actual_prices: Optional[pd.Series] = None):
        """
        Visualize Monte Carlo simulations.
        
        Args:
            prices: Array of simulated price paths
            actual_prices: Optional Series of actual prices for comparison
        """
        # Ensure prices are time_horizon x num_simulations
        if prices.shape[0] < prices.shape[1]:
            prices = prices.T
        
        # Create time index
        time_index = np.arange(prices.shape[0])
        
        # Plot simulations
        plt.figure(figsize=(12, 6))
        
        # Plot all simulation paths
        for i in range(prices.shape[1]):
            plt.plot(time_index, prices[:, i], color='blue', alpha=0.1)
        
        # Plot mean path
        mean_path = np.mean(prices, axis=1)
        plt.plot(time_index, mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Plot actual prices if provided
        if actual_prices is not None:
            plt.plot(time_index[:len(actual_prices)], actual_prices, color='green', linewidth=2, label='Actual Prices')
        
        plt.title(f"Monte Carlo Simulation ({self.model.replace('_', ' ').title()})")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.legend()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f"monte_carlo_{self.model}.png"))
        plt.close()
    
    def run_simulation(self, historical_prices: pd.Series, start_date: Optional[str] = None, 
                     output_filename: Optional[str] = None, actual_prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run the complete simulation process.
        
        Args:
            historical_prices: Historical price series for parameter estimation
            start_date: Optional start date (default: today)
            output_filename: Optional base filename for output files
            actual_prices: Optional Series of actual prices for comparison
        
        Returns:
            Dictionary of simulation results
        """
        logger.info(f"Running Monte Carlo simulation with {self.num_simulations} simulations")
        
        # Simulate price paths
        paths = self.simulate_price_paths(historical_prices)
        
        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(
            paths, self.confidence_intervals
        )
        
        # Calculate statistics
        statistics = self.calculate_statistics(paths)
        
        # Calculate Value at Risk
        var = self.calculate_value_at_risk(paths, 0.95)
        
        # Calculate Expected Shortfall
        es = self.calculate_expected_shortfall(paths, 0.95)
        
        # Create DataFrames
        dataframes = self.create_dataframe(paths, start_date)
        
        # Save simulations if filename provided
        if output_filename:
            self.save_simulations(dataframes, output_filename)
        else:
            self.save_simulations(dataframes, f"simulated_{self.model}")
        
        # Visualize simulations
        self.visualize_simulations(paths, actual_prices)
        
        # Return results dictionary
        return {
            'paths': paths,
            'confidence_intervals': confidence_intervals,
            'statistics': statistics,
            'var': var,
            'es': es
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monte Carlo price simulator')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--process', type=str, choices=['gbm', 'mean_reversion', 'jump_diffusion', 
                                                     'bull_market', 'bear_market', 'sideways', 'high_volatility'],
                        help='Simulation process to use')
    parser.add_argument('--initial-price', type=float, default=100.0,
                        help='Initial price')
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD) for the simulation')
    parser.add_argument('--output-filename', type=str,
                        help='Base filename for output files')
    parser.add_argument('--experiment', type=str,
                        help='Experiment name for output directory')
    
    return parser.parse_args()

def main():
    """Main function to run Monte Carlo simulation."""
    args = parse_args()
    
    # Setup logging
    log_file = f"monte_carlo_{args.experiment}.log" if args.experiment else "monte_carlo.log"
    setup_logger(log_file)
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override config with command line arguments
    if args.process:
        config['simulation']['process'] = args.process
    
    if args.experiment:
        config['simulation']['experiment_name'] = args.experiment
    
    # Create simulator
    simulator = MonteCarloSimulator(args.config)
    
    # Run simulation
    simulator.run_simulation(
        historical_prices=pd.Series(np.random.rand(100)),  # Replace with actual historical prices
        start_date=args.start_date,
        output_filename=args.output_filename
    )

if __name__ == "__main__":
    main()
