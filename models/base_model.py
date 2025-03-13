#!/usr/bin/env python
"""
Base model class for the trading system.

This module defines the base class for all models (statistical and ML/DL).
"""
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize the base model.
        
        Args:
            name: Model name
            params: Model parameters
        """
        self.name = name
        self.params = params if params is not None else {}
        self.model = None
        self.feature_columns = []
        self.target_column = "close"  # Default to 'close' if not specified
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Fit the model to training data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Array of predictions
        """
        pass
    
    def evaluate(self, data: pd.DataFrame, true_values: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            data: Test data
            true_values: Optional true values to compare against
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet. Attempting to evaluate.")
        
        # Use provided true values or fallback to target column
        if true_values is None:
            if self.target_column not in data.columns:
                logger.warning(f"Target column '{self.target_column}' not found. Using 'close'.")
                true_values = data['close'].values
            else:
                true_values = data[self.target_column].values
        
        # Get predictions
        y_pred = self.predict(data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_values, y_pred)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Directional Accuracy (for regression tasks)
        try:
            directional_accuracy = np.mean((y_true * y_pred) > 0)
        except Exception:
            directional_accuracy = np.nan
        
        # Return metrics
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save(self, directory: str):
        """
        Save the model to disk.
        
        Args:
            directory: Directory to save the model to
        """
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        model_path = os.path.join(directory, f"{self.name}.pkl")
        meta_path = os.path.join(directory, f"{self.name}_meta.pkl")
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        meta = {
            'params': self.params,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_fitted': self.is_fitted
        }
        
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, directory: str) -> bool:
        """
        Load the model from disk.
        
        Args:
            directory: Directory to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        model_path = os.path.join(directory, f"{self.name}.pkl")
        meta_path = os.path.join(directory, f"{self.name}_meta.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            logger.error(f"Model files not found at {directory}")
            return False
        
        try:
            # Load the model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            
            self.params = meta.get('params', {})
            self.feature_columns = meta.get('feature_columns', [])
            self.target_column = meta.get('target_column', 'close')
            self.is_fitted = meta.get('is_fitted', False)
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            str: Name of the model
        """
        return self.name
