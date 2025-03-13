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

class StrategyGridSearch:
    """Grid search for optimizing trading strategy parameters."""
    
    def __init__(self, config_path: str, experiment_dir: Optional[str] = None):
        """
        Initialize the grid search with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            experiment_dir: Optional experiment directory for outputs
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.experiment_dir = experiment_dir
        
        # If experiment_dir is provided, save all outputs there
        if experiment_dir:
            self.output_dir = os.path.join(experiment_dir, 'grid_search_results')
        else:
            self.output_dir = os.path.join(
                self.config.get('experiment', {}).get('base_output_dir', './output'),
                'grid_search_results'
            )
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize backtester
        self.backtester = Backtester(config_path)
    
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
    
    def _generate_parameter_grid(self, strategy_name: str, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all possible combinations of parameters.
        
        Args:
            strategy_name: Name of the strategy
            param_grid: Dictionary with parameter names as keys and lists of possible values
            
        Returns:
            List of parameter dictionaries, each representing one combination
        """
        # Get all parameter names
        param_names = list(param_grid.keys())
        
        # Generate all combinations of parameter values
        param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
        
        # Convert to list of dictionaries
        param_dicts = []
        for values in param_values:
            param_dict = {name: value for name, value in zip(param_names, values)}
            param_dicts.append(param_dict)
        
        return param_dicts
    
    def _evaluate_strategy(
        self, strategy_name: str, params: Dict[str, Any], data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate a strategy with specific parameters on the given data.
        
        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Dictionary with performance metrics
        """
        strategy_class = self._get_strategy_class(strategy_name)
        strategy = strategy_class(strategy_name, params)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Run backtest
        metrics = self.backtester.backtest(data, signals)
        
        return metrics
    
    def run_grid_search(
        self, 
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        metrics_to_maximize: List[str] = ['sharpe_ratio', 'total_return'],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Run grid search for a strategy on the given data.
        
        Args:
            strategy_name: Name of the strategy
            param_grid: Dictionary with parameter names as keys and lists of possible values
            data: DataFrame with OHLCV and indicator data
            metrics_to_maximize: List of metrics to maximize in order of priority
            top_n: Number of top performing parameter sets to return
            
        Returns:
            DataFrame with parameter combinations and performance metrics
        """
        logger.info(f"Starting grid search for {strategy_name}")
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Data shape: {data.shape}")
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_grid(strategy_name, param_grid)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        # Evaluate each parameter combination
        results = []
        
        for params in tqdm(param_combinations, desc=f"Grid search for {strategy_name}"):
            try:
                metrics = self._evaluate_strategy(strategy_name, params, data)
                
                # Combine parameters and metrics
                result = {**params, **metrics}
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {strategy_name} with params {params}: {str(e)}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by metrics in order of priority
        for metric in reversed(metrics_to_maximize):
            results_df = results_df.sort_values(by=metric, ascending=False)
        
        # Save all results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f"{strategy_name}_grid_search_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Saved all grid search results to {results_path}")
        
        # Get top N results
        top_results = results_df.head(top_n)
        
        # Save top results
        top_results_path = os.path.join(self.output_dir, f"{strategy_name}_top_{top_n}_{timestamp}.csv")
        top_results.to_csv(top_results_path, index=False)
        
        logger.info(f"Saved top {top_n} results to {top_results_path}")
        
        # Log best result
        best_params = {k: top_results.iloc[0][k] for k in param_grid.keys()}
        best_metrics = {
            metric: top_results.iloc[0][metric] 
            for metric in metrics_to_maximize if metric in top_results.columns
        }
        
        logger.info(f"Best parameters for {strategy_name}: {best_params}")
        logger.info(f"Best metrics: {best_metrics}")
        
        return results_df
    
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
                strategy_name=strategy_name,
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
        comparison_path = os.path.join(self.output_dir, f"strategy_comparison_{timestamp}.csv")
        comparison_df.to_csv(comparison_path)
        
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

# Add alias for StrategyGridSearch for backward compatibility
GridSearch = StrategyGridSearch

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
    grid_search = StrategyGridSearch(args.config, experiment_dir)
    
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
            strategy_name=args.strategy,
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
