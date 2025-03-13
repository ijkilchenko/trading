# Trading System

A comprehensive trading system for algorithmic trading, featuring data acquisition, processing, model training, strategy backtesting, and live trading capabilities. The system is designed to be modular, configurable, and extensively tested.

## Features

- **Data Processing**: Download and process market data from various sources
- **Technical Analysis**: Comprehensive set of technical indicators
- **Strategy Development**: Modular framework for implementing trading strategies
- **Backtesting**: Robust backtesting engine with transaction costs and slippage
- **Risk Management**: Advanced risk management and position sizing
- **Visualization**: Tools for visualizing trading results and market data
- **Testing**: Comprehensive test suite with coverage reporting

## Project Structure

```
trading/
├── backtesting/        # Backtesting engine and utilities
├── config/            # Configuration files
├── data/              # Data acquisition and processing
├── models/            # Trading models and algorithms
├── risk_management/   # Risk management tools
├── simulation/        # Monte Carlo and other simulations
├── strategies/        # Trading strategy implementations
├── tests/            # Test suite
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
├── trading/          # Live trading implementation
├── utils/            # Utility functions
├── visualization/    # Data visualization tools
├── main.py           # Main entry point
└── run_tests.py      # Test runner
```

## Installation

### Prerequisites

- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading.git
cd trading
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Testing

The project includes a comprehensive test suite with both unit and integration tests. The test runner (`run_tests.py`) provides detailed test results and coverage reports.

### Running Tests

Run all tests:
```bash
python run_tests.py
```

Run specific test types:
```bash
python run_tests.py --unit          # Run only unit tests
python run_tests.py --integration   # Run only integration tests
```

Additional test options:
```bash
python run_tests.py --verbose       # Detailed test output
python run_tests.py --coverage      # Generate coverage report (requires coverage package)
python run_tests.py --fail-fast     # Stop on first failure
python run_tests.py --module data   # Test specific module
python run_tests.py --output-dir test_reports  # Specify output directory for reports
python run_tests.py --no-color      # Disable colored output
```

### Alternative Testing with pytest

You can also run tests directly with pytest:

```bash
# Run all tests
python -m pytest

# Run specific test files
python -m pytest tests/unit/data/test_data_processor.py

# Run with verbosity
python -m pytest -v
```

## Usage

### Main Commands

The system provides a unified interface through `main.py`:

```bash
python main.py [command] [options]
```

Available commands:

1. Download market data:
```bash
python main.py download --symbols BTC-USD ETH-USD --start-date 2023-01-01 --end-date 2023-01-31
```

2. Process data:
```bash
python main.py process --symbols BTC-USD --output ./processed_data/btc_processed.pkl
```

3. Train models:
```bash
python main.py train --model ARIMA --data-path ./processed_data/btc_processed.pkl --output-dir ./models
```

4. Run backtesting:
```bash
python main.py backtest --strategy MovingAverageCrossover --data-path ./processed_data/btc_processed.pkl --output-dir ./backtest_results
```

5. Start trading:
```bash
python main.py trade --mode paper --strategy RSIThreshold
```

6. Run Monte Carlo simulation:
```bash
python main.py simulate --process gbm --initial-price 100.0 --output-dir ./simulation_results
```

7. Generate visualizations:
```bash
python main.py visualize --type backtest --data-path ./backtest_results/equity_curve.csv --output-dir ./visualizations
```

### Running Components Directly

You can also run individual components directly:

1. Download data:
```bash
python -m data.download_data --config ./config/config.yaml --symbols BTCUSDT ETHUSDT --start-date 2023-01-01
```

2. Process data:
```bash
python -m data.data_processor --config ./config/config.yaml --symbol BTCUSDT --output-dir ./processed_data
```

3. Run backtesting:
```bash
python -m backtesting.backtester --config ./config/config.yaml --data-path ./processed_data/btcusdt_processed.pkl --strategy MovingAverageCrossover --output-dir ./backtest_results
```

## Configuration

The system is configured through YAML files in the `config` directory. The default configuration file is `config/config.yaml`.

Key configuration areas include:

- Data sources and parameters
- Trading strategy parameters
- Risk management settings
- Backtesting configuration
- Live trading settings

Example configuration:
```yaml
data:
  binance:
    base_url: https://api.binance.us/api/v3
    symbols: [BTCUSDT, ETHUSDT]
    start_date: 2023-01-01
    end_date: 2023-01-31
    interval: 1m
    retries: 3
    timeout: 10
    rate_limit: 1200
  database:
    path: ./market_data.db
    table_prefix: binance_
  validation:
    check_missing_bars: true
    check_zero_volume: true
    max_missing_bars_pct: 5

processing:
  train_val_test_split:
    train: 0.7
    val: 0.15
    test: 0.15
  indicators:
    - name: SMA
      params:
        - window: 10
        - window: 20
        - window: 50
    - name: RSI
      params:
        - window: 14
    - name: MACD
      params:
        - fast: 12
          slow: 26
          signal: 9

strategies:
  - name: MovingAverageCrossover
    params:
      fast_ma: 10
      slow_ma: 30
      ma_type: sma
  - name: RSIThreshold
    params:
      rsi_period: 14
      overbought: 70
      oversold: 30

backtesting:
  initial_capital: 10000.0
  position_sizing: fixed
  fixed_position_size: 1000.0
  fees:
    maker: 0.001
    taker: 0.001
  slippage: 0.001
  output_dir: ./backtest_results
```

## Development

### Adding New Components

1. **New Strategy**:
   - Create new class in `strategies/strategy_implementations.py`
   - Implement the `generate_signals` method
   - Add unit tests in `tests/unit/strategies/`

2. **New Model**:
   - Add model class in appropriate file in `models/` directory
   - Implement `fit` and `predict` methods
   - Add unit tests in `tests/unit/models/`

3. **New Technical Indicator**:
   - Add to `data/technical_indicators.py`
   - Include validation and testing
   - Add unit tests in `tests/unit/data/`

### Code Quality

- Run tests before committing changes
- Ensure test coverage for new code
- Follow PEP 8 style guidelines
- Use type hints for better code clarity
