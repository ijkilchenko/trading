#!/usr/bin/env python
"""
Deep learning models for the trading system.

This module implements various neural network architectures using PyTorch.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class MLPModel(BaseModel):
    """Multilayer Perceptron model for time series forecasting."""
    
    def __init__(self, name: str, params: Dict):
        """
        Initialize the MLP model.
        
        Args:
            name: Model name
            params: Model parameters
        """
        super().__init__(name, params)
        self.hidden_layers = params.get('hidden_layers', [64, 32])
        self.activation = params.get('activation', 'relu')
        self.dropout = params.get('dropout', 0.2)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.batch_size = params.get('batch_size', 32)
        self.epochs = params.get('epochs', 100)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = StandardScaler()
    
    def _build_model(self, input_size: int):
        """
        Build the MLP model.
        
        Args:
            input_size: Number of input features
        """
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size
        
        # Output layer (regression)
        layers.append(nn.Linear(prev_size, 1))
        
        return nn.Sequential(*layers)
    
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Fit the MLP model to training data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Dictionary with training metrics
        """
        # Default to predicting future close price if target not specified
        if not self.target_column:
            self.target_column = 'future_close_1'
        
        # Select features (exclude target and timestamp-related columns)
        exclude_cols = ['timestamp', 'close_time', 'number_of_trades']
        exclude_cols.extend([col for col in train_data.columns if col.startswith('future_')])
        
        self.feature_columns = [col for col in train_data.columns 
                               if col not in exclude_cols and col != self.target_column]
        
        # Prepare data
        X_train = train_data[self.feature_columns].values
        y_train = train_data[self.target_column].values.reshape(-1, 1)
        
        X_val = val_data[self.feature_columns].values
        y_val = val_data[self.target_column].values.reshape(-1, 1)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Build model
        input_size = X_train.shape[1]
        self.model = self._build_model(input_size).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        try:
            for epoch in range(self.epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, "
                               f"Train Loss: {train_loss:.6f}, "
                               f"Val Loss: {val_loss:.6f}")
            
            # Load best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            self.fitted = True
            
            # Calculate metrics
            metrics = {
                'training_success': True,
                'train_loss': train_losses,
                'val_loss': val_losses,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': best_val_loss
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training MLP model: {e}")
            return {
                'training_success': False,
                'error': str(e)
            }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the MLP model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Array of predictions
        """
        if not self.fitted or not self.model:
            logger.error("Model not fitted yet")
            return np.array([])
        
        try:
            # Prepare features
            X = data[self.feature_columns].values
            
            # Scale features
            X = self.scaler.transform(X)
            
            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).cpu().numpy()
            
            return y_pred.flatten()
            
        except Exception as e:
            logger.error(f"Error making MLP predictions: {e}")
            return np.array([np.nan] * len(data))


class LSTMModel(BaseModel):
    """LSTM model for time series forecasting."""
    
    def __init__(self, name: str, params: Dict):
        """
        Initialize the LSTM model.
        
        Args:
            name: Model name
            params: Model parameters
        """
        super().__init__(name, params)
        self.hidden_size = params.get('hidden_size', 64)
        self.num_layers = params.get('num_layers', 2)
        self.dropout = params.get('dropout', 0.2)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.batch_size = params.get('batch_size', 32)
        self.epochs = params.get('epochs', 100)
        self.sequence_length = params.get('sequence_length', 30)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = StandardScaler()
    
    def _build_model(self, input_size: int):
        """
        Build the LSTM model.
        
        Args:
            input_size: Number of input features
        """
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(LSTMNet, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                # Initialize hidden state and cell state
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                # Forward propagate LSTM
                out, _ = self.lstm(x, (h0, c0))
                
                # We only need the output from the last time step
                out = self.fc(out[:, -1, :])
                
                return out
        
        return LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
    
    def _prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM model.
        
        Args:
            data: Feature data
            target: Target data
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i+self.sequence_length]
            label = target[i+self.sequence_length]
            sequences.append(seq)
            targets.append(label)
        
        return np.array(sequences), np.array(targets).reshape(-1, 1)
    
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Fit the LSTM model to training data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Dictionary with training metrics
        """
        # Default to predicting future close price if target not specified
        if not self.target_column:
            self.target_column = 'future_close_1'
        
        # Select features (exclude target and timestamp-related columns)
        exclude_cols = ['timestamp', 'close_time', 'number_of_trades']
        exclude_cols.extend([col for col in train_data.columns if col.startswith('future_')])
        
        self.feature_columns = [col for col in train_data.columns 
                               if col not in exclude_cols and col != self.target_column]
        
        # Prepare data
        X_train = train_data[self.feature_columns].values
        y_train = train_data[self.target_column].values
        
        X_val = val_data[self.feature_columns].values
        y_val = val_data[self.target_column].values
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq)
        X_val_tensor = torch.FloatTensor(X_val_seq)
        y_val_tensor = torch.FloatTensor(y_val_seq)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Build model
        input_size = X_train.shape[1]
        self.model = self._build_model(input_size).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        try:
            for epoch in range(self.epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, "
                               f"Train Loss: {train_loss:.6f}, "
                               f"Val Loss: {val_loss:.6f}")
            
            # Load best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            self.fitted = True
            
            # Calculate metrics
            metrics = {
                'training_success': True,
                'train_loss': train_losses,
                'val_loss': val_losses,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': best_val_loss
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {
                'training_success': False,
                'error': str(e)
            }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the LSTM model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Array of predictions
        """
        if not self.fitted or not self.model:
            logger.error("Model not fitted yet")
            return np.array([])
        
        try:
            # Prepare features
            X = data[self.feature_columns].values
            
            # Scale features
            X = self.scaler.transform(X)
            
            # Initialize output array with NaNs for the first sequence_length entries
            predictions = np.full(len(data), np.nan)
            
            # Make predictions for each sequence
            for i in range(self.sequence_length, len(X)):
                seq = X[i-self.sequence_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                
                predictions[i] = pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            return np.array([np.nan] * len(data))


class CNNModel(BaseModel):
    """1D CNN model for time series forecasting."""
    
    def __init__(self, name: str, params: Dict):
        """
        Initialize the CNN model.
        
        Args:
            name: Model name
            params: Model parameters
        """
        super().__init__(name, params)
        self.filters = params.get('filters', [64, 128, 64])
        self.kernel_sizes = params.get('kernel_sizes', [3, 3, 3])
        self.dropout = params.get('dropout', 0.2)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.batch_size = params.get('batch_size', 32)
        self.epochs = params.get('epochs', 100)
        self.sequence_length = params.get('sequence_length', 30)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = StandardScaler()
        self.fitted = False
        self.model = None
    
    def _build_model(self, input_size: int):
        """
        Build the CNN model.
        
        Args:
            input_size: Number of input features
        """
        class CNNNet(nn.Module):
            def __init__(self, input_size, filters, kernel_sizes, dropout, sequence_length=30):
                super(CNNNet, self).__init__()
                
                # Input shape: [batch_size, sequence_length, input_size]
                # For 1D CNN, we need to reshape to [batch_size, input_size, sequence_length]
                # to match PyTorch's Conv1d expectations
                
                self.cnn_layers = nn.ModuleList()
                in_channels = input_size
                self.sequence_length = sequence_length
                
                for i, (out_channels, kernel_size) in enumerate(zip(filters, kernel_sizes)):
                    self.cnn_layers.append(nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=kernel_size,
                        padding=kernel_size // 2  # Same padding
                    ))
                    self.cnn_layers.append(nn.ReLU())
                    self.cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                    self.cnn_layers.append(nn.Dropout(dropout))
                    in_channels = out_channels
                
                # Calculate the size after convolutions and pooling
                # This is approximate and may need adjustment
                final_length = self.sequence_length // (2 ** len(filters))
                if final_length == 0:
                    final_length = 1
                
                self.fc = nn.Linear(filters[-1] * final_length, 1)
            
            def forward(self, x):
                # x shape: [batch_size, sequence_length, input_size]
                # Transpose to [batch_size, input_size, sequence_length]
                x = x.transpose(1, 2)
                
                # Apply CNN layers
                for layer in self.cnn_layers:
                    x = layer(x)
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # Fully connected layer
                x = self.fc(x)
                
                return x
        
        model = CNNNet(input_size, self.filters, self.kernel_sizes, self.dropout, self.sequence_length)
        return model
    
    def _prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for CNN model.
        
        Args:
            data: Feature data
            target: Target data
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i+self.sequence_length]
            label = target[i+self.sequence_length]
            sequences.append(seq)
            targets.append(label)
        
        return np.array(sequences), np.array(targets).reshape(-1, 1)
    
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Fit the CNN model to training data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Dictionary with training metrics
        """
        # Default to predicting future close price if target not specified
        if not self.target_column:
            self.target_column = 'future_close_1'
        
        # Select features (exclude target and timestamp-related columns)
        exclude_cols = ['timestamp', 'close_time', 'number_of_trades']
        exclude_cols.extend([col for col in train_data.columns if col.startswith('future_')])
        
        self.feature_columns = [col for col in train_data.columns 
                               if col not in exclude_cols and col != self.target_column]
        
        # Prepare data
        X_train = train_data[self.feature_columns].values
        y_train = train_data[self.target_column].values
        
        X_val = val_data[self.feature_columns].values
        y_val = val_data[self.target_column].values
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq)
        X_val_tensor = torch.FloatTensor(X_val_seq)
        y_val_tensor = torch.FloatTensor(y_val_seq)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Build model
        input_size = X_train.shape[1]
        self.model = self._build_model(input_size).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        try:
            for epoch in range(self.epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                
                train_loss /= len(train_loader.dataset)
                train_losses.append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                
                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, "
                               f"Train Loss: {train_loss:.6f}, "
                               f"Val Loss: {val_loss:.6f}")
            
            # Load best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            self.fitted = True
            
            # Calculate metrics
            metrics = {
                'training_success': True,
                'train_loss': train_losses,
                'val_loss': val_losses,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': best_val_loss
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return {
                'training_success': False,
                'error': str(e)
            }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the CNN model.
        
        Args:
            data: Data to predict on
            
        Returns:
            Array of predictions
        """
        if not self.fitted or not self.model:
            logger.error("Model not fitted yet")
            return np.array([])
        
        try:
            # Prepare features
            X = data[self.feature_columns].values
            
            # Scale features
            X = self.scaler.transform(X)
            
            # Initialize output array with NaNs for the first sequence_length entries
            predictions = np.full(len(data), np.nan)
            
            # Make predictions for each sequence
            for i in range(self.sequence_length, len(X)):
                seq = X[i-self.sequence_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                
                predictions[i] = pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making CNN predictions: {e}")
            return np.array([np.nan] * len(data))
