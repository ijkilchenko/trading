#!/usr/bin/env python
"""
Logger utility for the trading system.

This module provides consistent logging setup across the trading system.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import tempfile
import gzip
import shutil
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform color support
colorama.init()

def compress_log_file(source_file, backup_count):
    """
    Compress log files using gzip.
    
    Args:
        source_file (str): Path to the source log file
        backup_count (int): Number of backup files to keep
    """
    import gzip
    import shutil
    
    # Check if file exists before attempting compression
    if not os.path.exists(source_file):
        return
    
    # Create compressed filename
    compressed_file = f"{source_file}.gz"
    
    # Compress the file
    with open(source_file, 'rb') as f_in:
        with gzip.open(compressed_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the original file
    os.remove(source_file)

def color_formatter(fmt, color=False):
    """
    Create a log formatter with optional color support.
    
    Args:
        fmt: Format string for the log message
        color: Whether to use color in the log message
    
    Returns:
        logging.Formatter: Configured log formatter
    """
    if not color:
        return logging.Formatter(fmt)
    
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: Fore.BLUE,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.MAGENTA
        }
        
        def format(self, record):
            log_message = super().format(record)
            color = self.COLORS.get(record.levelno, Fore.WHITE)
            return f"{color}{log_message}{Style.RESET_ALL}"
    
    return ColorFormatter(fmt)

class ExplicitStreamHandler(logging.StreamHandler):
    """
    A custom stream handler that explicitly calls write on the stream.
    """
    def emit(self, record):
        """
        Emit a record by writing to the stream.
        
        Overrides the default emit method to explicitly call write.
        """
        try:
            msg = self.format(record)
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()
        except Exception:
            self.handleError(record)

def setup_logger(log_file=None, level='INFO', log_to_console=True, color=False, 
                  retry=False, max_bytes=10*1024*1024, backup_count=5, 
                  log_format=None, compress=False):
    """
    Set up logger for the trading system.
    
    Args:
        log_file: Path to log file. If None, logging to file is disabled.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_console: Whether to log to console, defaults to True
        color: Whether to use color in console logging
        retry: Whether to retry file logging if permission error occurs
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        log_format: Custom log format string
        compress: Whether to compress rotated log files
    
    Returns:
        logging.Logger: Configured logger
    
    Raises:
        ValueError: If an invalid log level is provided
        IOError: If log file cannot be created or accessed
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        # Default to INFO if an invalid level is provided
        numeric_level = logging.INFO
    
    # Use default formatter if not provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler if log_to_console is True
    if log_to_console:
        console_handler = ExplicitStreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(color_formatter(log_format, color))
        root_logger.addHandler(console_handler)
    
    # Create file handler if log file is provided
    if log_file:
        file_handler = None
        
        # Attempt to create log file with multiple strategies
        attempts = [
            # 1. Original log file path
            log_file,
            # 2. Temporary file in system temp directory
            os.path.join(tempfile.gettempdir(), f'trading_log_{os.getpid()}.log'),
            # 3. Fallback to a completely random temp file
            tempfile.mktemp(prefix='trading_log_', suffix='.log')
        ]
        
        for attempt_file in attempts:
            try:
                # Ensure log directory exists
                log_dir = os.path.dirname(attempt_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # Create file handler with rotation
                file_handler = RotatingFileHandler(
                    attempt_file, 
                    maxBytes=max_bytes, 
                    backupCount=backup_count
                )
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(color_formatter(log_format, color))
                root_logger.addHandler(file_handler)
                
                # Add compression hook if enabled
                if compress:
                    def compression_hook(source, dest):
                        """Compress log files after rotation."""
                        if os.path.exists(source):
                            try:
                                compress_log_file(source, backup_count)
                            except Exception as e:
                                print(f"Error compressing log file: {e}")
                    
                    file_handler.rotator = compression_hook
                
                # If we successfully created a handler, break the loop
                break
            
            except (PermissionError, IOError) as e:
                # If it's the last attempt, raise an exception
                if attempt_file == attempts[-1]:
                    # If retry is False or all attempts fail, raise an exception
                    if not retry:
                        print(f"Permission denied: Could not create log file {log_file}")
                        break
                    else:
                        # If retry is True and all attempts fail, raise an exception
                        raise IOError(f"Could not create log file after multiple attempts: {e}")
                
                # Continue to next attempt if this one fails
                continue
            
            except Exception as e:
                # For any other unexpected errors
                print(f"Error creating log file: {e}")
                break
    
    # Ensure at least one handler is present
    if not root_logger.handlers:
        console_handler = ExplicitStreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(color_formatter(log_format, color))
        root_logger.addHandler(console_handler)
    
    return root_logger
