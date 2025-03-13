#!/usr/bin/env python
"""
Grid search module for optimizing trading strategy parameters.

This module provides functionality to systematically test different combinations
of strategy parameters to identify the most profitable configurations.
"""
import argparse
import itertools
import logging
import os
import sys
import yaml
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from strategies.base_strategy import BaseStrategy
from backtesting.backtester import Backtester

logger = logging.getLogger(__name__)

class GridSearch:
    """Grid search for optimizing trading strategy parameters."""
    
    def __init__(self, config_path: str, experiment_dir: Optional[str] = None):
        """
        Initialize the grid search with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            experiment_dir: Optional experiment directory for outputs
        """
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            # Handle case where config is passed directly as a dict
            self.config = config_path
        
        self.experiment_dir = experiment_dir
        
        # If experiment_dir is provided, save all outputs there
        if experiment_dir:
            self.output_dir = os.path.join(experiment_dir, 'grid_search_results')
        else:
            output_base = self.config.get('outputs', {}).get('base_dir', '.')
            self.output_dir = os.path.join(output_base, 'grid_search_results')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract grid search configuration
        grid_search_config = self.config.get('grid_search', {})
        self.optimization_metric = grid_search_config.get('optimization_metric', 'total_return')
        self.n_jobs = grid_search_config.get('n_jobs', 1)
        
        # Initialize backtester
        self.backtester = Backtester(self.config)
    
    def _get_strategy_class(self, strategy_name: str) -> type:
        """
        Get strategy class by name.
        
        Args:
            strategy_name: Name of the strategy class
            
        Returns:
            Strategy class
        """
        # Import here to avoid circular imports
        from strategies.strategy_implementations import (
            MovingAverageCrossover, RSIThreshold, BollingerBreakout,
            MACDStrategy, SupportResistance
        )
        
        strategy_classes = {
            'MovingAverageCrossover': MovingAverageCrossover,
            'RSIThreshold': RSIThreshold,
            'BollingerBreakout': BollingerBreakout,
            'MACDStrategy': MACDStrategy,
            'SupportResistance': SupportResistance,
        }
        
        if strategy_name not in strategy_classes:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return strategy_classes[strategy_name]
    
    def _generate_parameter_grid(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters for grid search.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter dictionaries, one for each combination
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of parameter dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def _evaluate_parameters(self, strategy: BaseStrategy, params: Dict[str, Any], 
                           data: pd.DataFrame, symbol: str = "BTCUSDT",
                           backtester = None) -> Dict[str, float]:
        """
        Evaluate a set of parameters for a strategy.
        
        Args:
            strategy: Strategy to evaluate
            params: Parameters to set
            data: Data to evaluate on
            symbol: Symbol to use for evaluation
            backtester: Optional backtester to use (uses self.backtester if None)
            
        Returns:
            Dictionary of performance metrics
        """
        # Set parameters
        strategy.set_parameters(params)
        
        # Run backtest
        bt = backtester if backtester is not None else Backtester(self.config)
        results = bt.run_backtest(strategy, data, symbol)
        
        # Return evaluation metrics
        return results['metrics']
        
    def save_results(self, results: Union[pd.DataFrame, List[Dict]], strategy_name: str, symbol: str) -> str:
        """
        Save grid search results to disk.
        
        Args:
            results: DataFrame or list of dictionaries with grid search results
            strategy_name: Name of the strategy
            symbol: Symbol used for backtesting
            
        Returns:
            Path to the saved results file
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate output file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{strategy_name}_{symbol}_{timestamp}.csv"
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert list to DataFrame if needed
        if isinstance(results, list):
            # If results is a list of dicts with nested dicts, flatten them
            flat_results = []
            for result in results:
                flat_result = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_result[f"{key}_{subkey}"] = subvalue
                    else:
                        flat_result[key] = value
                flat_results.append(flat_result)
            results_df = pd.DataFrame(flat_results)
        else:
            results_df = results
        
        # Save results
        results_df.to_csv(output_path, index=False)
        logger.info(f"Grid search results saved to {output_path}")
        
        return output_path
    
    def run_grid_search(
        self, 
        strategy_or_name: Union[str, BaseStrategy],
        param_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        symbol: str = "BTCUSDT",
        backtester = None,
        metrics_to_maximize: List[str] = None
    ) -> pd.DataFrame:
        """
        Run grid search for a strategy on the given data.
        
        Args:
            strategy_or_name: Name of the strategy or strategy instance
            param_grid: Dictionary with parameter names as keys and lists of possible values
            data: DataFrame with OHLCV and indicator data
            symbol: Symbol for backtesting
            backtester: Optional backtester instance to use
            metrics_to_maximize: List of metrics to maximize in order of priority
            
        Returns:
            DataFrame with parameter combinations and performance metrics
        """
        if metrics_to_maximize is None:
            metrics_to_maximize = ['total_return', 'sharpe_ratio']
            
        # Handle both strategy name string and strategy instance
        if isinstance(strategy_or_name, str):
            strategy_name = strategy_or_name
            strategy_instance = None
        else:
            strategy_name = strategy_or_name.get_name()
            strategy_instance = strategy_or_name
        
        logger.info(f"Starting grid search for {strategy_name}")
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Data shape: {data.shape}")
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_grid(param_grid)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Evaluate each parameter combination
        results = []
        
        for params in tqdm(param_combinations, desc=f"Grid search for {strategy_name}"):
            try:
                # Create strategy instance if needed
                if strategy_instance is None:
                    strategy = self._get_strategy_class(strategy_name)(strategy_name, params)
                else:
                    strategy = strategy_instance
                    strategy.set_parameters(params)
                
                metrics = self._evaluate_parameters(strategy, params, data, symbol, backtester)
                
                # Combine parameters and metrics
                result = {**params, **metrics}
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {strategy_name} with params {params}: {str(e)}")
        
        # Convert to DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Sort by metrics in order of priority
            for metric in reversed(metrics_to_maximize):
                if metric in results_df.columns:
                    results_df = results_df.sort_values(by=metric, ascending=False)
            
            return results_df
        else:
            # Return empty DataFrame with expected columns
            columns = list(param_grid.keys()) + list(metrics_to_maximize)
            return pd.DataFrame(columns=columns)
    
    def run_multi_strategy_grid_search(
        self,
        strategies_param_grids: Dict[str, Dict[str, List[Any]]],
        data: pd.DataFrame,
        metrics_to_maximize: List[str] = ['sharpe_ratio', 'total_return'],
        top_n: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Run grid search for multiple strategies on the given data.
        
        Args:
            strategies_param_grids: Dictionary with strategy names as keys and parameter grids as values
            data: DataFrame with OHLCV and indicator data
            metrics_to_maximize: List of metrics to maximize in order of priority
            top_n: Number of top performing parameter sets to return for each strategy
            
        Returns:
            Dictionary with strategy names as keys and DataFrames with results as values
        """
        logger.info(f"Starting multi-strategy grid search")
        logger.info(f"Number of strategies: {len(strategies_param_grids)}")
        
        results = {}
        
        for strategy_name, param_grid in strategies_param_grids.items():
            logger.info(f"Running grid search for {strategy_name}")
            
            strategy_results = self.run_grid_search(
                strategy_or_name=strategy_name,
                param_grid=param_grid,
                data=data,
                metrics_to_maximize=metrics_to_maximize,
                top_n=top_n
            )
            
            results[strategy_name] = strategy_results
        
        # Compare best results across strategies
        best_results = {}
        
        for strategy_name, results_df in results.items():
            # Get best row for this strategy
            best_row = results_df.iloc[0].to_dict()
            best_results[strategy_name] = best_row
        
        # Convert to DataFrame for easier comparison
        comparison_df = pd.DataFrame.from_dict(best_results, orient='index')
        
        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = self.save_results(comparison_df, "strategy_comparison", "comparison")
        
        logger.info(f"Saved strategy comparison to {comparison_path}")
        
        # Log best overall strategy
        best_strategy = None
        best_metric_value = -float('inf')
        
        for strategy, metrics in best_results.items():
            metric_value = metrics[metrics_to_maximize[0]]
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_strategy = strategy
        
        if best_strategy:
            logger.info(f"Best overall strategy: {best_strategy}")
            logger.info(f"Best {metrics_to_maximize[0]}: {best_metric_value}")
            logger.info(f"Parameters: {best_results[best_strategy]}")
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Grid search for trading strategies')
    
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment', type=str,
                      help='Experiment name for versioning outputs')
    parser.add_argument('--data-path', type=str, required=True,
                      help='Path to processed data file')
    parser.add_argument('--strategy', type=str,
                      help='Strategy to optimize (if not specified, will optimize all strategies)')
    parser.add_argument('--param-grid', type=str,
                      help='Path to parameter grid JSON file (overrides config)')
    parser.add_argument('--metrics', type=str, nargs='+', default=['sharpe_ratio', 'total_return'],
                      help='Metrics to maximize in order of priority')
    parser.add_argument('--top-n', type=int, default=10,
                      help='Number of top performing parameter sets to return')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    setup_logger()
    
    # Create experiment directory if needed
    experiment_dir = None
    if args.experiment:
        from utils.experiment import setup_experiment_dir
        experiment_dir = setup_experiment_dir(args.experiment)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data = pd.read_pickle(args.data_path)
    
    # Initialize grid search
    grid_search = GridSearch(args.config, experiment_dir)
    
    # Load parameter grid
    if args.param_grid:
        with open(args.param_grid, 'r') as file:
            param_grids = json.load(file)
    else:
        # Use default parameter grids from config
        param_grids = {}
        
        for strategy_config in grid_search.config.get('strategies', []):
            strategy_name = strategy_config['name']
            strategy_params = strategy_config['params']
            
            # Create grid for this strategy
            param_grid = {}
            
            for param_name, param_value in strategy_params.items():
                # If not specified in args.param_grid, use a range around the default value
                if isinstance(param_value, (int, float)):
                    # For numeric parameters, create a range
                    if isinstance(param_value, int):
                        param_grid[param_name] = [
                            max(1, param_value - 2),
                            param_value - 1,
                            param_value,
                            param_value + 1,
                            param_value + 2
                        ]
                    else:  # float
                        param_grid[param_name] = [
                            param_value * 0.8,
                            param_value * 0.9,
                            param_value,
                            param_value * 1.1,
                            param_value * 1.2
                        ]
                elif isinstance(param_value, str):
                    # For string parameters, just use the default
                    param_grid[param_name] = [param_value]
            
            param_grids[strategy_name] = param_grid
    
    # Run grid search
    if args.strategy:
        if args.strategy not in param_grids:
            logger.error(f"Strategy {args.strategy} not found in parameter grids")
            return
        
        param_grid = param_grids[args.strategy]
        
        results = grid_search.run_grid_search(
            strategy_or_name=args.strategy,
            param_grid=param_grid,
            data=data,
            metrics_to_maximize=args.metrics,
            top_n=args.top_n
        )
    else:
        # Run for all strategies
        results = grid_search.run_multi_strategy_grid_search(
            strategies_param_grids=param_grids,
            data=data,
            metrics_to_maximize=args.metrics,
            top_n=args.top_n
        )

if __name__ == "__main__":
    main()
