#!/usr/bin/env python
"""
Main entry point for the trading system.

This script provides a unified interface to run different components:
- Data download
- Data processing
- Model training
- Backtesting
- Live trading
- Monte Carlo simulation
- Visualization
"""
import argparse
import logging
import os
import sys
import yaml
import shutil
from datetime import datetime
from pathlib import Path

from utils.logger import setup_logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading System')
    
    # Common arguments
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment', type=str,
                      help='Experiment name for versioning outputs')
    parser.add_argument('--log-level', type=str, default='INFO',
                      help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download data command
    download_parser = subparsers.add_parser('download', help='Download market data')
    download_parser.add_argument('--symbols', type=str, nargs='+',
                               help='Symbols to download (overrides config)')
    download_parser.add_argument('--start-date', type=str,
                               help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', type=str,
                               help='End date (YYYY-MM-DD)')
    
    # Process data command
    process_parser = subparsers.add_parser('process', help='Process market data')
    process_parser.add_argument('--symbols', type=str, nargs='+',
                              help='Symbols to process (overrides config)')
    process_parser.add_argument('--features', type=str, nargs='+',
                              help='Features to calculate (overrides config)')
    process_parser.add_argument('--output', type=str, 
                              help='Output path for processed data')
    
    # Train models command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', type=str, 
                            help='Model to train (overrides config)')
    train_parser.add_argument('--data-path', type=str,
                            help='Path to processed data')
    train_parser.add_argument('--output-dir', type=str,
                            help='Directory to save trained models')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--strategy', type=str, required=True,
                               help='Strategy to backtest')
    backtest_parser.add_argument('--data-path', type=str, required=True,
                               help='Path to processed data')
    backtest_parser.add_argument('--start-date', type=str,
                               help='Start date for backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str,
                               help='End date for backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--output-dir', type=str,
                               help='Directory to save backtest results')
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Run trading')
    trade_parser.add_argument('--mode', type=str, choices=['paper', 'live'], 
                            default='paper', help='Trading mode')
    trade_parser.add_argument('--strategy', type=str, required=True,
                            help='Strategy to use for trading')
    
    # Monte Carlo simulation command
    monte_carlo_parser = subparsers.add_parser('simulate', help='Run Monte Carlo simulation')
    monte_carlo_parser.add_argument('--process', type=str, 
                                  choices=['gbm', 'mean_reversion', 'jump_diffusion', 
                                         'bull_market', 'bear_market', 'sideways', 
                                         'high_volatility'],
                                  help='Simulation process to use')
    monte_carlo_parser.add_argument('--initial-price', type=float, default=100.0,
                                  help='Initial price')
    monte_carlo_parser.add_argument('--start-date', type=str,
                                  help='Start date for simulation (YYYY-MM-DD)')
    monte_carlo_parser.add_argument('--output-dir', type=str,
                                  help='Directory to save simulation results')
    monte_carlo_parser.add_argument('--compare-data', type=str,
                                  help='Path to actual data for comparison')
    
    # Visualization command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    visualize_parser.add_argument('--type', type=str, required=True,
                                choices=['training', 'backtest', 'indicators', 'monte_carlo'],
                                help='Type of visualization to generate')
    visualize_parser.add_argument('--data-path', type=str, required=True,
                                help='Path to data for visualization')
    visualize_parser.add_argument('--output-dir', type=str,
                                help='Directory to save visualizations')
    visualize_parser.add_argument('--include', type=str, nargs='+',
                                help='Specific elements to include in visualization')
    
    return parser.parse_args()

def setup_experiment_dir(config, args):
    """
    Set up experiment directory for versioned outputs.
    
    Args:
        config: Configuration dictionary
        args: Command-line arguments
        
    Returns:
        Updated configuration with experiment directory
    """
    # Create base directories if they don't exist
    output_base = config.get('output_dir', './output')
    os.makedirs(output_base, exist_ok=True)
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.experiment:
        experiment_name = f"{args.experiment}_{timestamp}"
    elif args.command:
        experiment_name = f"{args.command}_{timestamp}"
    else:
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(output_base, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories for different output types
    os.makedirs(os.path.join(experiment_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'backtest_results'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'simulations'), exist_ok=True)
    
    # Save a copy of the configuration
    config_copy_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_copy_path, 'w') as file:
        yaml.dump(config, file)
    
    # Update configuration with experiment directory
    config['experiment'] = {
        'name': args.experiment or experiment_name,
        'directory': experiment_dir,
        'timestamp': timestamp,
        'data_dir': os.path.join(experiment_dir, 'data'),
        'models_dir': os.path.join(experiment_dir, 'models'),
        'backtest_dir': os.path.join(experiment_dir, 'backtest_results'),
        'logs_dir': os.path.join(experiment_dir, 'logs'),
        'visualizations_dir': os.path.join(experiment_dir, 'visualizations'),
        'simulations_dir': os.path.join(experiment_dir, 'simulations')
    }
    
    return config

def main():
    """Main function to run the trading system."""
    args = parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Set up experiment directory if specified
    if args.experiment or (args.command and args.command not in ['download']):
        config = setup_experiment_dir(config, args)
        
        # Update log file to be in the experiment directory
        log_file = os.path.join(config['experiment']['logs_dir'], f"{args.command}.log")
    else:
        # Default log file
        log_file = f"trading_system_{args.command}.log"
    
    # Setup logging
    logger = setup_logger(log_file, args.log_level)
    logger.info(f"Loaded configuration from {args.config}")
    
    if 'experiment' in config:
        logger.info(f"Running experiment: {config['experiment']['name']}")
        logger.info(f"Experiment directory: {config['experiment']['directory']}")
    
    # Execute command
    if args.command == 'download':
        from data.download_data import main as download_main
        sys.argv = [sys.argv[0]]  # Reset argv for the imported main function
        if args.symbols:
            sys.argv.extend(['--symbols'] + args.symbols)
        if args.start_date:
            sys.argv.extend(['--start-date', args.start_date])
        if args.end_date:
            sys.argv.extend(['--end-date', args.end_date])
        sys.argv.extend(['--config', args.config])
        download_main()
    
    elif args.command == 'process':
        from data.data_processor import main as process_main
        sys.argv = [sys.argv[0]]
        if args.symbols:
            sys.argv.extend(['--symbols'] + args.symbols)
        if args.features:
            sys.argv.extend(['--features'] + args.features)
        
        # Use experiment directory for output if available
        output_path = args.output
        if not output_path and 'experiment' in config:
            output_path = os.path.join(config['experiment']['data_dir'], 'processed_data.pkl')
            sys.argv.extend(['--output', output_path])
        elif output_path:
            sys.argv.extend(['--output', output_path])
            
        sys.argv.extend(['--config', args.config])
        process_main()
    
    elif args.command == 'train':
        from models.train_models import main as train_main
        sys.argv = [sys.argv[0]]
        if args.model:
            sys.argv.extend(['--model', args.model])
        
        # Use experiment directory for data path and output if available
        data_path = args.data_path
        if not data_path and 'experiment' in config:
            data_path = os.path.join(config['experiment']['data_dir'], 'processed_data.pkl')
            # Check if the file exists, otherwise use default
            if not os.path.exists(data_path):
                data_path = args.data_path
        
        if data_path:
            sys.argv.extend(['--data-path', data_path])
            
        output_dir = args.output_dir
        if not output_dir and 'experiment' in config:
            output_dir = config['experiment']['models_dir']
            sys.argv.extend(['--output-dir', output_dir])
        elif output_dir:
            sys.argv.extend(['--output-dir', output_dir])
            
        sys.argv.extend(['--config', args.config])
        train_main()
    
    elif args.command == 'backtest':
        from backtesting.backtester import main as backtest_main
        sys.argv = [sys.argv[0]]
        sys.argv.extend(['--strategy', args.strategy])
        
        # Handle data path
        if 'experiment' in config and not args.data_path:
            # Try to find processed data in the experiment directory
            data_path = os.path.join(config['experiment']['data_dir'], 'processed_data.pkl')
            if os.path.exists(data_path):
                sys.argv.extend(['--data-path', data_path])
            else:
                sys.argv.extend(['--data-path', args.data_path])
        else:
            sys.argv.extend(['--data-path', args.data_path])
            
        if args.start_date:
            sys.argv.extend(['--start-date', args.start_date])
        if args.end_date:
            sys.argv.extend(['--end-date', args.end_date])
        
        # Handle output directory
        output_dir = args.output_dir
        if not output_dir and 'experiment' in config:
            output_dir = config['experiment']['backtest_dir']
            sys.argv.extend(['--output-dir', output_dir])
        elif output_dir:
            sys.argv.extend(['--output-dir', output_dir])
            
        sys.argv.extend(['--config', args.config])
        backtest_main()
    
    elif args.command == 'trade':
        from trading.trader import main as trade_main
        sys.argv = [sys.argv[0]]
        sys.argv.extend(['--mode', args.mode])
        sys.argv.extend(['--strategy', args.strategy])
        
        # If we have an experiment, pass it to the trader
        if 'experiment' in config:
            sys.argv.extend(['--output-dir', config['experiment']['directory']])
            
        sys.argv.extend(['--config', args.config])
        sys.argv.extend(['--log-level', args.log_level])
        trade_main()
    
    elif args.command == 'simulate':
        from simulation.monte_carlo import main as simulate_main
        sys.argv = [sys.argv[0]]
        
        if args.process:
            sys.argv.extend(['--process', args.process])
        if args.initial_price:
            sys.argv.extend(['--initial-price', str(args.initial_price)])
        if args.start_date:
            sys.argv.extend(['--start-date', args.start_date])
        
        # Handle output directory
        output_dir = args.output_dir
        if not output_dir and 'experiment' in config:
            output_dir = config['experiment']['simulations_dir']
            sys.argv.extend(['--output-dir', output_dir])
        elif output_dir:
            sys.argv.extend(['--output-dir', output_dir])
            
        # Add experiment name if available
        if 'experiment' in config:
            sys.argv.extend(['--experiment', config['experiment']['name']])
            
        sys.argv.extend(['--config', args.config])
        simulate_main()
    
    elif args.command == 'visualize':
        from utils.visualization import Visualizer
        
        # Determine output directory
        output_dir = args.output_dir
        if not output_dir and 'experiment' in config:
            output_dir = config['experiment']['visualizations_dir']
        elif not output_dir:
            output_dir = './visualizations'
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizer
        experiment_name = config.get('experiment', {}).get('name', None)
        visualizer = Visualizer(output_dir, experiment_name)
        
        # Generate visualizations based on type
        if args.type == 'training':
            import pandas as pd
            
            # Load training metrics
            try:
                metrics_df = pd.read_csv(args.data_path)
                
                # Plot training loss
                if 'train_loss' in metrics_df.columns and 'val_loss' in metrics_df.columns:
                    visualizer.plot_training_loss(
                        metrics_df['train_loss'].tolist(),
                        metrics_df['val_loss'].tolist(),
                        metrics_df['epoch'].tolist() if 'epoch' in metrics_df.columns else None,
                        title="Training Loss Curve",
                        filename="training_loss.png"
                    )
                
                # Plot other metrics
                metrics_columns = [col for col in metrics_df.columns if col not in ['epoch', 'train_loss', 'val_loss']]
                if metrics_columns:
                    metrics_dict = {col: metrics_df[col].tolist() for col in metrics_columns}
                    visualizer.plot_metrics(
                        metrics_dict,
                        metrics_df['epoch'].tolist() if 'epoch' in metrics_df.columns else None,
                        title="Training Metrics",
                        filename="training_metrics.png"
                    )
                    
                logger.info(f"Generated training visualizations in {output_dir}")
                
            except Exception as e:
                logger.error(f"Error generating training visualizations: {e}")
        
        elif args.type == 'backtest':
            import pandas as pd
            
            # Load backtest results
            try:
                # Try to find equity curve and trades in the data path
                equity_curve_path = os.path.join(args.data_path, 'equity_curve.csv')
                trades_path = os.path.join(args.data_path, 'trades.csv')
                
                if os.path.exists(equity_curve_path):
                    equity_curve = pd.read_csv(equity_curve_path, index_col=0, parse_dates=True)
                    
                    # Plot equity curve
                    visualizer.plot_equity_curve(
                        equity_curve,
                        title="Equity Curve",
                        filename="equity_curve.png"
                    )
                    
                    # Plot drawdown
                    visualizer.plot_drawdown(
                        equity_curve,
                        title="Drawdown",
                        filename="drawdown.png"
                    )
                    
                    # Plot monthly returns
                    visualizer.plot_monthly_returns(
                        equity_curve,
                        title="Monthly Returns",
                        filename="monthly_returns.png"
                    )
                
                if os.path.exists(trades_path):
                    trades = pd.read_csv(trades_path)
                    
                    # Plot trade distribution
                    visualizer.plot_trade_distribution(
                        trades,
                        title="Trade P&L Distribution",
                        filename="trade_distribution.png"
                    )
                
                logger.info(f"Generated backtest visualizations in {output_dir}")
                
            except Exception as e:
                logger.error(f"Error generating backtest visualizations: {e}")
        
        elif args.type == 'indicators':
            import pandas as pd
            
            # Load data with indicators
            try:
                data = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
                
                # Define indicator groups based on column names
                indicators = {}
                
                # Group columns into indicator types
                if args.include:
                    for indicator in args.include:
                        if indicator.lower() in ['ma', 'ema', 'sma']:
                            indicators['moving_averages'] = [col for col in data.columns if any(
                                ma_type in col.lower() for ma_type in ['ma', 'ema', 'sma'])]
                        elif indicator.lower() == 'bollinger':
                            indicators['bollinger_bands'] = [col for col in data.columns if 'bollinger' in col.lower()]
                        elif indicator.lower() in ['rsi', 'stoch']:
                            indicators['oscillators'] = [col for col in data.columns if any(
                                osc in col.lower() for osc in ['rsi', 'stoch'])]
                        elif indicator.lower() == 'macd':
                            indicators['macd'] = [col for col in data.columns if 'macd' in col.lower()]
                else:
                    # Auto-detect indicators
                    indicators['moving_averages'] = [col for col in data.columns if any(
                        ma_type in col.lower() for ma_type in ['ma', 'ema', 'sma'])]
                    indicators['bollinger_bands'] = [col for col in data.columns if 'bollinger' in col.lower()]
                    indicators['oscillators'] = [col for col in data.columns if any(
                        osc in col.lower() for osc in ['rsi', 'stoch'])]
                    indicators['macd'] = [col for col in data.columns if 'macd' in col.lower()]
                
                # Plot technical indicators
                visualizer.plot_technical_indicators(
                    data,
                    indicators,
                    title="Price and Technical Indicators",
                    filename="technical_indicators.png"
                )
                
                logger.info(f"Generated indicator visualizations in {output_dir}")
                
            except Exception as e:
                logger.error(f"Error generating indicator visualizations: {e}")
        
        elif args.type == 'monte_carlo':
            import pandas as pd
            import numpy as np
            import pickle
            
            # Load Monte Carlo simulation results
            try:
                # Check if data_path is a pickle file with simulations
                if args.data_path.endswith('.pkl'):
                    with open(args.data_path, 'rb') as f:
                        simulations = pickle.load(f)
                    
                    # Convert to numpy array if it's a list of DataFrames
                    if isinstance(simulations, list) and isinstance(simulations[0], pd.DataFrame):
                        # Extract closing prices
                        prices = np.array([df['close'].values for df in simulations])
                    elif isinstance(simulations, np.ndarray):
                        prices = simulations
                    else:
                        logger.error("Unsupported simulation data format")
                        sys.exit(1)
                    
                    # Load actual prices if specified
                    actual_prices = None
                    if args.include and os.path.exists(args.include[0]):
                        actual_df = pd.read_csv(args.include[0], index_col=0, parse_dates=True)
                        if 'close' in actual_df.columns:
                            actual_prices = actual_df['close']
                    
                    # Plot Monte Carlo simulations
                    visualizer.plot_monte_carlo_simulations(
                        prices,
                        actual_prices,
                        title="Monte Carlo Price Simulations",
                        filename="monte_carlo_simulations.png"
                    )
                    
                    logger.info(f"Generated Monte Carlo visualizations in {output_dir}")
                
                else:
                    logger.error(f"Invalid data path format for Monte Carlo visualization: {args.data_path}")
            
            except Exception as e:
                logger.error(f"Error generating Monte Carlo visualizations: {e}")
    
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
