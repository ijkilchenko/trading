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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
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
python run_tests.py --coverage      # Generate coverage report
python run_tests.py --fail-fast     # Stop on first failure
python run_tests.py --module data   # Test specific module
```

### Test Reports

Test results and coverage reports are saved in the `test_reports` directory. The reports include:
- Overall test statistics
- Module-level test results
- Error analysis
- Code coverage metrics

## Usage

### Main Commands

The system provides a unified interface through `main.py`:

```bash
python main.py [command] [options]
```

Available commands:

1. Download market data:
```bash
python main.py download --symbols BTC-USD ETH-USD --start-date 2023-01-01
```

2. Process data:
```bash
python main.py process --symbols BTC-USD
```

3. Run backtesting:
```bash
python main.py backtest --strategy MovingAverageCrossover
```

4. Start trading:
```bash
python main.py trade --mode paper --strategy RSIStrategy
```

5. Run simulation:
```bash
python main.py simulate --strategy DualMovingAverage
```

## Configuration

The system is configured through YAML files in the `config` directory. Key configuration areas include:

- Data sources and parameters
- Trading strategy parameters
- Risk management settings
- Backtesting configuration
- Live trading settings

Example configuration:
```yaml
data:
  symbols: ["BTC-USD", "ETH-USD"]
  timeframe: "1h"
  source: "binance"

strategy:
  name: "RSIStrategy"
  parameters:
    rsi_period: 14
    oversold: 30
    overbought: 70

risk_management:
  max_position_size: 0.1
  stop_loss_pct: 0.02
  take_profit_pct: 0.05
```

## Development

### Adding New Components

1. **New Strategy**:
   - Create new class in `strategies/`
   - Implement required methods
   - Add unit tests in `tests/unit/strategies/`

2. **New Model**:
   - Add model class in `models/`
   - Implement training and prediction logic
   - Add unit tests in `tests/unit/models/`

3. **New Indicator**:
   - Add to `data/technical_indicators.py`
   - Include validation and testing
   - Add unit tests in `tests/unit/data/`

### Code Quality

- Run tests before committing changes
- Ensure test coverage for new code
- Follow PEP 8 style guidelines
- Use type hints for better code clarity
