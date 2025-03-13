#!/usr/bin/env python
"""
Model training script for the trading system.

This script handles model training, evaluation, and checkpoint saving.
"""
import argparse
import logging
import os
import pickle
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from models.statistical_models import ARIMAModel, GARCHModel
from models.dl_models import MLPModel, LSTMModel, CNNModel

logger = logging.getLogger(__name__)

def create_model(model_type: str, model_name: str, params: Dict):
    """
    Create a model instance based on the type.
    
    Args:
        model_type: Type of model (ARIMA, GARCH, MLP, LSTM, CNN)
        model_name: Name for the model instance
        params: Model parameters
        
    Returns:
        Model instance
    """
    if model_type == 'ARIMA':
        return ARIMAModel(model_name, params)
    elif model_type == 'GARCH':
        return GARCHModel(model_name, params)
    elif model_type == 'MLP':
        return MLPModel(model_name, params)
    elif model_type == 'LSTM':
        return LSTMModel(model_name, params)
    elif model_type == 'CNN':
        return CNNModel(model_name, params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def plot_loss_curves(train_loss: List[float], val_loss: List[float], output_dir: str, model_name: str):
    """
    Plot training and validation loss curves.
    
    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        output_dir: Directory to save plot
        model_name: Name of the model
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_loss_curves.png'))
    plt.close()

def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                    output_dir: str, model_name: str, title: str = 'Model Predictions'):
    """
    Plot actual vs. predicted values.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        output_dir: Directory to save plot
        model_name: Name of the model
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Filter out NaN values from predictions
    mask = ~np.isnan(predicted)
    indices = np.arange(len(actual))[mask]
    
    plt.plot(indices, actual[mask], label='Actual', alpha=0.7)
    plt.plot(indices, predicted[mask], label='Predicted', alpha=0.7)
    
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_predictions.png'))
    plt.close()

def train_model(model_type: str, model_params: Dict, data: Dict, target_column: str, 
               output_dir: str, experiment_name: str = None):
    """
    Train a model and save it to disk.
    
    Args:
        model_type: Type of model
        model_params: Model parameters
        data: Dictionary with train/val/test data
        target_column: Target column name
        output_dir: Directory to save model and plots
        experiment_name: Optional experiment name
        
    Returns:
        Dictionary with training results
    """
    # Create model instance
    model_name = model_type
    if experiment_name:
        model_name = f"{model_type}_{experiment_name}"
    
    logger.info(f"Training {model_name} model")
    
    try:
        model = create_model(model_type, model_name, model_params)
        
        # Set target column
        model.target_column = target_column
        
        # Train model
        train_results = model.fit(data['train_data'], data['val_data'])
        
        if not train_results['training_success']:
            logger.error(f"Failed to train {model_name}: {train_results.get('error', 'Unknown error')}")
            return train_results
        
        # Evaluate on test data
        test_metrics = model.evaluate(data['test_data'])
        train_results['test_metrics'] = test_metrics
        
        # Make predictions on test data
        test_predictions = model.predict(data['test_data'])
        
        # Save model
        model_dir = os.path.join(output_dir, 'models', model_name)
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
        
        # Plot loss curves if available
        if 'train_loss' in train_results and 'val_loss' in train_results:
            plot_loss_curves(
                train_results['train_loss'],
                train_results['val_loss'],
                os.path.join(output_dir, 'plots'),
                model_name
            )
        
        # Plot predictions
        if target_column in data['test_data']:
            plot_predictions(
                data['test_data'][target_column].values,
                test_predictions,
                os.path.join(output_dir, 'plots'),
                model_name,
                f'{model_name} - Actual vs. Predicted'
            )
        
        # Save predictions
        predictions_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        predictions_df = data['test_data'].copy()
        predictions_df[f'{model_name}_pred'] = test_predictions
        predictions_df.to_csv(os.path.join(predictions_dir, f'{model_name}_predictions.csv'))
        
        logger.info(f"Model {model_name} trained successfully")
        logger.info(f"Test metrics: {test_metrics}")
        
        return {
            'model_name': model_name,
            'training_success': True,
            'train_results': train_results,
            'test_metrics': test_metrics
        }
        
    except Exception as e:
        logger.error(f"Error training {model_name}: {e}")
        return {
            'model_name': model_name,
            'training_success': False,
            'error': str(e)
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model configuration file (overrides config.yaml)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to processed data file')
    parser.add_argument('--target-column', type=str, default='future_return_1',
                        help='Target column to predict')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--model-type', type=str, 
                        choices=['ARIMA', 'GARCH', 'MLP', 'LSTM', 'CNN', 'all'],
                        default='all',
                        help='Type of model to train')
    
    return parser.parse_args()

def main():
    """Main function to run the model training."""
    args = parse_args()
    
    # Setup logging
    log_file = f"train_models_{args.experiment}.log" if args.experiment else "train_models.log"
    setup_logger(log_file)
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override with model-specific config if provided
    if args.model_config:
        with open(args.model_config, 'r') as file:
            model_config = yaml.safe_load(file)
        config['models'] = model_config
    
    # Create output directory
    output_dir = args.output_dir
    if args.experiment:
        output_dir = os.path.join(output_dir, args.experiment)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        with open(args.data_path, 'rb') as file:
            data = pickle.load(file)
        
        logger.info(f"Loaded data from {args.data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Train models
    all_results = {}
    
    # Statistical models
    if args.model_type in ['ARIMA', 'all'] and 'statistical' in config['models']:
        for model_config in config['models']['statistical']:
            if model_config['name'] == 'ARIMA':
                result = train_model(
                    'ARIMA',
                    model_config['params'],
                    data,
                    args.target_column,
                    output_dir,
                    args.experiment
                )
                all_results['ARIMA'] = result
    
    if args.model_type in ['GARCH', 'all'] and 'statistical' in config['models']:
        for model_config in config['models']['statistical']:
            if model_config['name'] == 'GARCH':
                result = train_model(
                    'GARCH',
                    model_config['params'],
                    data,
                    args.target_column,
                    output_dir,
                    args.experiment
                )
                all_results['GARCH'] = result
    
    # Machine learning models
    if args.model_type in ['MLP', 'all'] and 'machine_learning' in config['models']:
        for model_config in config['models']['machine_learning']:
            if model_config['name'] == 'MLP':
                result = train_model(
                    'MLP',
                    model_config['params'],
                    data,
                    args.target_column,
                    output_dir,
                    args.experiment
                )
                all_results['MLP'] = result
    
    if args.model_type in ['LSTM', 'all'] and 'machine_learning' in config['models']:
        for model_config in config['models']['machine_learning']:
            if model_config['name'] == 'LSTM':
                result = train_model(
                    'LSTM',
                    model_config['params'],
                    data,
                    args.target_column,
                    output_dir,
                    args.experiment
                )
                all_results['LSTM'] = result
    
    if args.model_type in ['CNN', 'all'] and 'machine_learning' in config['models']:
        for model_config in config['models']['machine_learning']:
            if model_config['name'] == 'CNN':
                result = train_model(
                    'CNN',
                    model_config['params'],
                    data,
                    args.target_column,
                    output_dir,
                    args.experiment
                )
                all_results['CNN'] = result
    
    # Save overall results
    results_path = os.path.join(output_dir, 'training_results.pkl')
    with open(results_path, 'wb') as file:
        pickle.dump(all_results, file)
    
    # Print summary
    logger.info("Training completed. Results summary:")
    for model_name, result in all_results.items():
        success = result.get('training_success', False)
        metrics = result.get('test_metrics', {})
        
        if success:
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"{model_name}: Success - {metrics_str}")
        else:
            error = result.get('error', 'Unknown error')
            logger.info(f"{model_name}: Failed - {error}")

if __name__ == "__main__":
    main()
