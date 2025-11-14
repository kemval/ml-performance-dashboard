"""
Automated Test Runner
Runs all tests and generates comprehensive HTML report
"""

import pytest
import sys
import os
from datetime import datetime
import json


def run_all_tests():
    """Run all tests and generate reports"""
    
    print("=" * 70)
    print("🧪 ML MODEL PERFORMANCE - AUTOMATED TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Test configuration
    test_dirs = [
        'tests/test_utils.py',
        'tests/test_data_pipeline.py',
        'tests/test_model_validation.py'
    ]
    
    # Generate HTML report
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    print("📋 Running Test Suites:")
    print("  ✓ Unit Tests (test_utils.py)")
    print("  ✓ Integration Tests (test_data_pipeline.py)")
    print("  ✓ Model Validation Tests (test_model_validation.py)")
    print()
    
    # Run pytest with detailed output
    args = [
        '-v',                          # Verbose
        '-s',                          # Show print statements
        '--tb=short',                  # Short traceback format
        f'--html={report_file}',      # HTML report
        '--self-contained-html',       # Single file report
        '--continue-on-collection-errors',  # Continue on errors
    ] + test_dirs
    
    print("🚀 Executing tests...\n")
    
    # Run tests
    exit_code = pytest.main(args)
    
    print()
    print("=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
        print(f"📄 Detailed report: {report_file}")
    elif exit_code == 1:
        print("❌ SOME TESTS FAILED")
        print(f"📄 Check detailed report: {report_file}")
    else:
        print(f"⚠️  Test execution completed with code: {exit_code}")
        print(f"📄 Check detailed report: {report_file}")
    
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return exit_code


def run_specific_test_suite(suite_name):
    """Run a specific test suite"""
    
    suite_map = {
        'unit': 'tests/test_utils.py',
        'integration': 'tests/test_data_pipeline.py',
        'validation': 'tests/test_model_validation.py'
    }
    
    if suite_name not in suite_map:
        print(f"❌ Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(suite_map.keys())}")
        return 1
    
    print(f"🧪 Running {suite_name.upper()} tests...")
    print()
    
    exit_code = pytest.main(['-v', '-s', suite_map[suite_name]])
    
    return exit_code


def list_all_tests():
    """List all available tests"""
    print("=" * 70)
    print("📋 AVAILABLE TESTS")
    print("=" * 70)
    print()
    
    pytest.main(['--collect-only', 'tests/'])
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'list':
            list_all_tests()
        elif command in ['unit', 'integration', 'validation']:
            exit_code = run_specific_test_suite(command)
            sys.exit(exit_code)
        elif command == 'help':
            print("🧪 ML Model Test Runner")
            print()
            print("Usage:")
            print("  python test_runner.py           # Run all tests")
            print("  python test_runner.py unit      # Run unit tests only")
            print("  python test_runner.py integration   # Run integration tests only")
            print("  python test_runner.py validation    # Run model validation tests only")
            print("  python test_runner.py list      # List all available tests")
            print("  python test_runner.py help      # Show this help message")
        else:
            print(f"❌ Unknown command: {command}")
            print("Run 'python test_runner.py help' for usage information")
            sys.exit(1)
    else:
        # Run all tests by default
        exit_code = run_all_tests()
        sys.exit(exit_code)