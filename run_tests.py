#!/usr/bin/env python
"""
Test runner for the trading system.

This script discovers and runs all tests in the project, and produces a comprehensive
report of the results. It also creates coverage reports.
"""
import os
import sys
import unittest
import argparse
import logging
from datetime import datetime
import importlib
import re
from collections import Counter, defaultdict

# Add the project root to the path to allow imports
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
# Also add tests directory to the path
sys.path.insert(0, os.path.join(base_dir, 'tests'))

# Try to import coverage module, but don't fail if it's not installed
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREY = '\033[90m'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for the trading system")
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Run tests in verbose mode")
    
    parser.add_argument('-c', '--coverage', action='store_true',
                        help="Generate coverage reports")
    
    parser.add_argument('-u', '--unit', action='store_true',
                        help="Run only unit tests")
    
    parser.add_argument('-i', '--integration', action='store_true',
                        help="Run only integration tests")
    
    parser.add_argument('-o', '--output-dir', type=str, default='test_reports',
                        help="Directory to save test reports (default: test_reports)")
    
    parser.add_argument('-f', '--fail-fast', action='store_true',
                        help="Stop test run on first error or failure")
    
    parser.add_argument('-m', '--module', type=str, default=None,
                        help="Run tests only for a specific module (e.g., data, models)")
    
    parser.add_argument('--no-color', action='store_true',
                        help="Disable colored output")
    
    return parser.parse_args()

def setup_logging(output_dir):
    """Set up logging for the test runner."""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up logging
    log_file = os.path.join(output_dir, f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('test_runner')

def run_single_test_file(file_path):
    """Run a single test file using unittest."""
    # Extract the module path from the file path
    rel_path = os.path.relpath(file_path, base_dir)
    module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
    
    try:
        # Try to import the module
        module = importlib.import_module(module_path)
        # Load tests from the module
        return unittest.defaultTestLoader.loadTestsFromModule(module)
    except (ImportError, AttributeError) as e:
        logging.getLogger('test_runner').warning(f"Error importing {module_path}: {e}")
        return unittest.TestSuite()

def discover_tests(start_dir, pattern='test_*.py', module=None):
    """Discover tests to run."""
    logger = logging.getLogger('test_runner')
    
    if not os.path.exists(start_dir):
        logger.warning(f"Test directory does not exist: {start_dir}")
        return unittest.TestSuite()
    
    test_suite = unittest.TestSuite()
    
    if module:
        # If a specific module is requested, look for tests in that module
        module_dir = os.path.join(start_dir, module)
        if not os.path.exists(module_dir):
            logger.warning(f"Module directory not found: {module}")
            return unittest.TestSuite()
        
        logger.info(f"Discovering tests in module: {module}")
        
        # Find all test files in the module directory
        for root, _, files in os.walk(module_dir):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    test_suite.addTest(run_single_test_file(file_path))
    else:
        # Otherwise, discover all tests
        logger.info(f"Discovering tests in: {start_dir}")
        
        # Find all test files in the directory and subdirectories
        for root, _, files in os.walk(start_dir):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    test_suite.addTest(run_single_test_file(file_path))
    
    return test_suite

def custom_test_result():
    """Create a custom test result class to capture more detailed information."""
    class CustomTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_categories = defaultdict(list)
            self.error_counts = Counter()
            self.failure_counts = Counter()
            self.successes = []
            self.test_outcomes = {}  # Track outcome of each test
            
        def addSuccess(self, test):
            super().addSuccess(test)
            self.successes.append(test)
            test_module = test.__class__.__module__
            self.test_categories[test_module].append(('success', test))
            self.test_outcomes[test] = 'success'
        
        def addError(self, test, err):
            super().addError(test, err)
            test_module = test.__class__.__module__
            self.test_categories[test_module].append(('error', test, err))
            error_type = err[0].__name__
            self.error_counts[error_type] += 1
            self.test_outcomes[test] = 'error'
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            test_module = test.__class__.__module__
            self.test_categories[test_module].append(('failure', test, err))
            error_msg = str(err[1])
            self.failure_counts[error_msg] += 1
            self.test_outcomes[test] = 'failure'
        
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            test_module = test.__class__.__module__
            self.test_categories[test_module].append(('skipped', test, reason))
            self.test_outcomes[test] = 'skipped'
            
        def wasSuccessful(self):
            return len(self.failures) == 0 and len(self.errors) == 0
            
        @property
        def passed_tests(self):
            return len(self.successes)
            
        @property
        def failed_tests(self):
            return len(self.failures)
            
        @property
        def error_tests(self):
            return len(self.errors)
            
        @property
        def skipped_tests(self):
            return len(self.skipped)
    
    return CustomTestResult

def colorize(text, color, use_color=True):
    """Apply color to text if colored output is enabled."""
    if use_color:
        return f"{color}{text}{Colors.ENDC}"
    return text

def run_tests(test_suite, verbose=False, fail_fast=False, use_color=True):
    """
    Run the specified tests with configurable options.
    
    Args:
        test_suite (unittest.TestSuite): Test suite to run
        verbose (bool): Enable verbose output
        fail_fast (bool): Stop on first failure
        use_color (bool): Use color in output
    
    Returns:
        CustomTestResult: Test results
    """
    # Configure logging
    logger = logging.getLogger('test_runner')
    logger.info('Starting test runner...')
    
    # Configure test runner
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        failfast=fail_fast,
        resultclass=custom_test_result()
    )
    
    # Run tests and capture results
    result = runner.run(test_suite)
    
    # Print test summary
    print_test_summary(result)
    
    return result

def print_test_summary(result):
    """
    Print a comprehensive summary of test results.
    
    Args:
        result: CustomTestResult object containing test results
    """
    # Calculate totals using our tracked outcomes
    total_tests = result.testsRun  # This is an integer
    passed = len(result.successes)
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    
    # Calculate pass rate
    pass_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    
    # Determine overall status color
    if pass_rate == 100:
        status_color = Colors.OKGREEN
        status_icon = "✅"
    elif pass_rate >= 80:
        status_color = Colors.WARNING
        status_icon = "⚠️"
    else:
        status_color = Colors.FAIL
        status_icon = "❌"
    
    # Initialize module test results tracking
    module_test_results = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0})
    
    # Process test results by module using test_categories
    for module, test_list in result.test_categories.items():
        for test_info in test_list:
            outcome = test_info[0]  # 'success', 'error', 'failure', or 'skipped'
            module_test_results[module]['total'] += 1
            
            if outcome == 'success':
                module_test_results[module]['passed'] += 1
            elif outcome == 'error':
                module_test_results[module]['errors'] += 1
            elif outcome == 'failure':
                module_test_results[module]['failed'] += 1
            elif outcome == 'skipped':
                module_test_results[module]['skipped'] += 1
    
    # Error analysis dictionary
    error_analysis = {
        'types': {},
        'modules': {},
        'detailed_errors': []
    }
    
    # Process errors
    for test, error_info in result.errors:
        try:
            # Get error type safely - it might be a tuple, string, or exception
            if isinstance(error_info, tuple) and len(error_info) > 0:
                if isinstance(error_info[0], type):
                    error_type = error_info[0].__name__
                else:
                    error_type = str(error_info[0])
            else:
                error_type = "UnknownError"
                
            error_message = str(error_info[1]) if isinstance(error_info, tuple) and len(error_info) > 1 else str(error_info)
            test_name = getattr(test, '_testMethodName', str(test))
            module_name = test.__class__.__module__
            
            # Update error counts
            error_analysis['types'][error_type] = error_analysis['types'].get(error_type, 0) + 1
            error_analysis['modules'][module_name] = error_analysis['modules'].get(module_name, 0) + 1
            
            # Store detailed error info
            error_analysis['detailed_errors'].append({
                'type': error_type,
                'module': module_name,
                'test': test_name,
                'message': error_message
            })
        except Exception as e:
            print(f"Error analyzing test error: {e}")
            # Add to unknown errors
            error_analysis['types']["UnknownError"] = error_analysis['types'].get("UnknownError", 0) + 1
    
    # Process failures
    for test, failure_info in result.failures:
        try:
            error_type = 'AssertionFailure'  # More specific than just 'Failure'
            error_message = str(failure_info)
            test_name = getattr(test, '_testMethodName', str(test))
            module_name = test.__class__.__module__
            
            # Update failure counts
            error_analysis['types'][error_type] = error_analysis['types'].get(error_type, 0) + 1
            error_analysis['modules'][module_name] = error_analysis['modules'].get(module_name, 0) + 1
            
            # Store detailed failure info
            error_analysis['detailed_errors'].append({
                'type': error_type,
                'module': module_name,
                'test': test_name,
                'message': error_message
            })
        except Exception as e:
            print(f"Error analyzing test failure: {e}")
            # Add to unknown failures
            error_analysis['types']["UnknownFailure"] = error_analysis['types'].get("UnknownFailure", 0) + 1
    
    # Print summary header with color
    print('\n' + '=' * 80)
    print(f'{status_color}{status_icon} TEST RESULTS SUMMARY {status_icon}{Colors.ENDC}'.center(80))
    print('=' * 80)
    
    # Module-level Test Results
    if module_test_results:
        print('\n📊 MODULE TEST RESULTS')
        sorted_modules = sorted(module_test_results.items(), key=lambda x: x[1]['total'], reverse=True)
        for module, results in sorted_modules:
            module_pass_rate = (results['passed'] / results['total']) * 100 if results['total'] > 0 else 0
            
            # Determine module status color
            if module_pass_rate == 100:
                module_color = Colors.OKGREEN
            elif module_pass_rate >= 80:
                module_color = Colors.WARNING
            else:
                module_color = Colors.FAIL
                
            print(f"{module_color}{module}: {results['passed']}/{results['total']} passed ({module_pass_rate:.1f}%), "
                  f"Failed: {results['failed']}, Errors: {results['errors']}, "
                  f"Skipped: {results['skipped']}{Colors.ENDC}")
    
    # Error Analysis
    if error_analysis['types']:
        print(f'\n{Colors.FAIL}🔍 ERROR ANALYSIS{Colors.ENDC}')
        print('\nError Types:')
        for error_type, count in sorted(error_analysis['types'].items(), key=lambda x: x[1], reverse=True):
            print(f'  {Colors.FAIL}{error_type}: {count}{Colors.ENDC}')
        
        print('\nMost Affected Modules:')
        for module, count in sorted(error_analysis['modules'].items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f'  {Colors.FAIL}{module}: {count} issues{Colors.ENDC}')
            
        # Show some error details
        if error_analysis['detailed_errors']:
            print('\nSample Error Messages:')
            for error in sorted(error_analysis['detailed_errors'], key=lambda x: x['type'])[:3]:
                message = error['message'].split('\n')[0][:100]  # First line, truncated
                print(f"  {Colors.FAIL}{error['type']} in {error['test']}: {message}...{Colors.ENDC}")
    
    # Print recommendations if there are failures
    if failed > 0 or errors > 0:
        print(f'\n{Colors.BOLD}🛠️ RECOMMENDED ACTIONS{Colors.ENDC}')
        if pass_rate < 100:
            print(f'{Colors.WARNING}1. Focus on fixing failed tests first{Colors.ENDC}')
            print(f'{Colors.WARNING}2. Investigate modules with lowest pass rates{Colors.ENDC}')
            print(f'{Colors.WARNING}3. Check for common error patterns{Colors.ENDC}')

    # Overall Statistics with color
    print('\n📊 OVERALL STATISTICS')
    print(f'Total Tests: {Colors.BOLD}{total_tests}{Colors.ENDC}')
    
    # Print pass rate with appropriate color
    pass_status = f'Passed: {passed}/{total_tests} ({pass_rate:.1f}%)'
    if pass_rate == 100:
        print(f'{Colors.OKGREEN}{pass_status} - Perfect! All tests passed!{Colors.ENDC}')
    elif pass_rate >= 80:
        print(f'{Colors.WARNING}{pass_status} - Good, but some tests need attention{Colors.ENDC}')
    else:
        print(f'{Colors.FAIL}{pass_status} - Significant failures detected{Colors.ENDC}')
    
    # Print failures and errors in red if they exist
    if failed > 0:
        print(f'{Colors.FAIL}Failed: {failed}{Colors.ENDC}')
    else:
        print(f'Failed: {failed}')
        
    if errors > 0:
        print(f'{Colors.FAIL}Errors: {errors}{Colors.ENDC}')
    else:
        print(f'Errors: {errors}')
    
    if skipped > 0:
        print(f'{Colors.WARNING}Skipped: {skipped}{Colors.ENDC}')
    else:
        print(f'Skipped: {skipped}')

    print('\n' + '=' * 80)

def analyze_test_errors(result):
    """Analyze and categorize test errors and failures."""
    error_analysis = {
        'types': {},
        'modules': {},
        'detailed_errors': []
    }
    
    def extract_test_info(test):
        """Extract test information safely."""
        try:
            # Try to get the test name and module
            test_name = getattr(test, '__name__', str(test))
            module_name = getattr(test, '__module__', 'unknown_module')
        except Exception:
            # Fallback to string representation if attributes are not available
            test_name = str(test)
            module_name = 'unknown_module'
        return test_name, module_name
    
    # Analyze errors
    for test, error_info in result.errors:
        try:
            error_type = type(error_info[0]).__name__
            error_message = str(error_info[1])
            
            test_name, module_name = extract_test_info(test)
            
            # Count error types
            error_analysis['types'][error_type] = error_analysis['types'].get(error_type, 0) + 1
            
            # Count errors by module
            error_analysis['modules'][module_name] = error_analysis['modules'].get(module_name, 0) + 1
            
            # Store detailed error information
            error_analysis['detailed_errors'].append({
                'test': test_name,
                'module': module_name,
                'type': error_type,
                'message': error_message
            })
        except Exception as e:
            # Log any unexpected errors during analysis
            print(f"Error analyzing test error: {e}")
    
    # Analyze failures
    for test, failure_info in result.failures:
        try:
            error_type = 'Failure'
            error_message = str(failure_info[1])
            
            test_name, module_name = extract_test_info(test)
            
            # Count failures
            error_analysis['types'][error_type] = error_analysis['types'].get(error_type, 0) + 1
            
            # Count failures by module
            error_analysis['modules'][module_name] = error_analysis['modules'].get(module_name, 0) + 1
            
            # Store detailed failure information
            error_analysis['detailed_errors'].append({
                'test': test_name,
                'module': module_name,
                'type': error_type,
                'message': error_message
            })
        except Exception as e:
            # Log any unexpected errors during analysis
            print(f"Error analyzing test failure: {e}")
    
    return error_analysis

def save_results(result, output_dir, use_color=True):
    """Save test results to file and print a detailed report."""
    logger = logging.getLogger('test_runner')
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save test results
    result_file = os.path.join(output_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Categorize errors and failures
    categories = {}
    for test, error in result.errors + result.failures:
        error_type = type(error[0]).__name__
        categories[error_type] = categories.get(error_type, [])
        categories[error_type].append((test, error))
    
    with open(result_file, 'w') as f:
        f.write("=== Trading System Test Results ===\n\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tests Run: {result.testsRun}\n")
        f.write(f"Passed: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Skipped: {len(result.skipped)}\n")
        f.write("\n")
        
        f.write("=== Problem Categories ===\n\n")
        for category, items in categories.items():
            if items:
                f.write(f"--- {category.replace('_', ' ').title()} ({len(items)}) ---\n")
                for test, details in items:
                    f.write(f"  * {test}: {details}\n")
                f.write("\n")
        
        f.write("=== Detailed Results ===\n\n")
        if result.failures:
            f.write("=== Failures ===\n")
            for test, tb in result.failures:
                f.write(f"\n--- {test} ---\n")
                f.write(f"{tb}\n")
        
        if result.errors:
            f.write("=== Errors ===\n")
            for test, tb in result.errors:
                f.write(f"\n--- {test} ---\n")
                f.write(f"{tb}\n")
        
        if result.skipped:
            f.write("=== Skipped ===\n")
            for test, reason in result.skipped:
                f.write(f"\n--- {test} ---\n")
                f.write(f"Reason: {reason}\n")
    
    logger.info(f"Test results saved to: {result_file}")
    return result_file

def run_with_coverage(args, test_suite):
    """Run tests with coverage."""
    logger = logging.getLogger('test_runner')
    
    if not COVERAGE_AVAILABLE:
        logger.warning("Coverage module not available. Please install it with 'pip install coverage'.")
        return None
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Configure coverage
    cov = coverage.Coverage(
        source=['data', 'models', 'strategies', 'backtesting', 
                'visualization', 'risk_management', 'simulation', 'utils'],
        omit=['*/__pycache__/*', '*/tests/*', '*/.venv/*'],
    )
    
    # Start coverage
    logger.info("Starting coverage measurement")
    cov.start()
    
    # Run tests
    result = run_tests(test_suite, args.verbose, args.fail_fast, not args.no_color)
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Generate coverage reports
    logger.info("Generating coverage reports")
    
    # HTML report
    html_dir = os.path.join(args.output_dir, 'coverage_html')
    cov.html_report(directory=html_dir)
    logger.info(f"HTML coverage report saved to: {html_dir}")
    
    # XML report for CI tools
    xml_file = os.path.join(args.output_dir, 'coverage.xml')
    cov.xml_report(outfile=xml_file)
    
    # Text report in the console
    print("\n=== Coverage Summary ===")
    total_coverage = cov.report()
    print(f"Total coverage: {total_coverage:.2f}%")
    
    return result

def main():
    """Run the test suite."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting test runner...")
    
    # Build test suite
    all_tests = unittest.TestSuite()
    
    # Discover unit tests
    if not args.integration:
        unit_tests_dir = os.path.join(base_dir, 'tests', 'unit')
        unit_tests = discover_tests(unit_tests_dir, module=args.module)
        logger.info(f"Discovered {unit_tests.countTestCases()} unit tests")
        all_tests.addTest(unit_tests)
    
    # Discover integration tests
    if not args.unit:
        integration_tests_dir = os.path.join(base_dir, 'tests', 'integration')
        integration_tests = discover_tests(integration_tests_dir, module=args.module)
        logger.info(f"Discovered {integration_tests.countTestCases()} integration tests")
        all_tests.addTest(integration_tests)
    
    # Check if we have any tests
    if all_tests.countTestCases() == 0:
        logger.error("No tests discovered")
        return 1
    
    # Run the tests
    if args.coverage and COVERAGE_AVAILABLE:
        result = run_with_coverage(args, all_tests)
    else:
        result = run_tests(all_tests, args.verbose, args.fail_fast, not args.no_color)
        
    # Save the results
    save_results(result, args.output_dir, not args.no_color)
    
    # Return appropriate exit code
    if result.wasSuccessful():
        logger.info("All tests passed!")
        return 0
    else:
        logger.info("Tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
