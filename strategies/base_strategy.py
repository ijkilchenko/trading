#!/usr/bin/env python
"""
Base strategy class for the trading system.

This module defines the base class for all trading strategies.
"""
import logging
import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters (default: empty dict)
        """
        self.name = name
        self.params = params if params is not None else {}
    
    def get_name(self) -> str:
        """
        Get the strategy name.
        
        Returns:
            Strategy name
        """
        return self.name
    
    def get_parameters(self) -> Dict:
        """
        Get the strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return self.params
    
    def set_parameters(self, params: Dict) -> None:
        """
        Set the strategy parameters.
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            Series with trading signals:
            1 for buy, -1 for sell, 0 for hold
        """
        pass
    
    def save(self, directory: str):
        """
        Save the strategy to disk.
        
        Args:
            directory: Directory to save the strategy
        """
        os.makedirs(directory, exist_ok=True)
        
        strategy_path = os.path.join(directory, f"{self.name}_strategy.pkl")
        
        # Save strategy parameters
        with open(strategy_path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'params': self.params
            }, f)
        
        logger.info(f"Strategy saved to {strategy_path}")
    
    def load(self, directory: str) -> bool:
        """
        Load the strategy from disk.
        
        Args:
            directory: Directory to load the strategy from
            
        Returns:
            True if successful, False otherwise
        """
        strategy_path = os.path.join(directory, f"{self.name}_strategy.pkl")
        
        if not os.path.exists(strategy_path):
            logger.error(f"Strategy file not found at {strategy_path}")
            return False
        
        try:
            # Load strategy parameters
            with open(strategy_path, 'rb') as f:
                strategy_dict = pickle.load(f)
            
            self.name = strategy_dict['name']
            self.params = strategy_dict['params']
            
            logger.info(f"Strategy loaded from {strategy_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
            return False
