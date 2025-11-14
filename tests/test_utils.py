"""
Unit tests for utility functions
Tests individual functions in isolation
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    prepare_dataframe,
    calculate_performance_change,
    get_best_model,
    detect_performance_degradation,
    get_performance_status
)


class TestPrepareDataframe:
    """Test data preparation functionality"""
    
    def test_prepare_dataframe_with_valid_data(self):
        """Test that valid data is converted to DataFrame correctly"""
        mock_data = [
            {
                'model_name': 'TestModel',
                'model_version': 'v1.0',
                'model_full_name': 'TestModel_v1.0',
                'evaluation_date': '2024-11-13',
                'dataset': 'test',
                'metrics': {
                    'accuracy': 0.85,
                    'precision': 0.83,
                    'recall': 0.84,
                    'f1_score': 0.835,
                    'inference_time_ms': 45.2,
                    'loss': 0.25
                },
                'metadata': {
                    'samples_evaluated': 10000,
                    'aws_s3_path': 's3://test/path'
                }
            }
        ]
        
        df = prepare_dataframe(mock_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'accuracy' in df.columns
        assert df['accuracy'].iloc[0] == 0.85
    
    def test_prepare_dataframe_with_empty_data(self):
        """Test handling of empty data"""
        df = prepare_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_dataframe_columns_exist(self):
        """Test that all expected columns are present"""
        mock_data = [
            {
                'model_name': 'TestModel',
                'model_version': 'v1.0',
                'model_full_name': 'TestModel_v1.0',
                'evaluation_date': '2024-11-13',
                'dataset': 'test',
                'metrics': {
                    'accuracy': 0.85,
                    'precision': 0.83,
                    'recall': 0.84,
                    'f1_score': 0.835,
                    'inference_time_ms': 45.2,
                    'loss': 0.25
                },
                'metadata': {
                    'samples_evaluated': 10000,
                    'aws_s3_path': 's3://test/path'
                }
            }
        ]
        
        df = prepare_dataframe(mock_data)
        
        expected_columns = [
            'model_name', 'model_version', 'accuracy', 'precision',
            'recall', 'f1_score', 'inference_time_ms', 'loss'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Column {col} missing from DataFrame"


class TestPerformanceChange:
    """Test performance change calculations"""
    
    def test_calculate_performance_increase(self):
        """Test calculation of performance improvement"""
        mock_df = pd.DataFrame({
            'model_full_name': ['Model_A', 'Model_A', 'Model_A'],
            'evaluation_date': pd.to_datetime(['2024-11-01', '2024-11-05', '2024-11-10']),
            'accuracy': [0.80, 0.85, 0.90]
        })
        
        change = calculate_performance_change(mock_df, 'Model_A', 'accuracy')
        
        assert change > 0, "Should detect performance increase"
        assert change == 12.5, f"Expected 12.5% increase, got {change}%"
    
    def test_calculate_performance_decrease(self):
        """Test calculation of performance degradation"""
        mock_df = pd.DataFrame({
            'model_full_name': ['Model_A', 'Model_A', 'Model_A'],
            'evaluation_date': pd.to_datetime(['2024-11-01', '2024-11-05', '2024-11-10']),
            'accuracy': [0.90, 0.85, 0.80]
        })
        
        change = calculate_performance_change(mock_df, 'Model_A', 'accuracy')
        
        assert change < 0, "Should detect performance decrease"
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        mock_df = pd.DataFrame({
            'model_full_name': ['Model_A'],
            'evaluation_date': pd.to_datetime(['2024-11-01']),
            'accuracy': [0.85]
        })
        
        change = calculate_performance_change(mock_df, 'Model_A', 'accuracy')
        
        assert change == 0, "Should return 0 for insufficient data"


class TestBestModel:
    """Test best model identification"""
    
    def test_get_best_model(self):
        """Test identification of best performing model"""
        mock_df = pd.DataFrame({
            'model_full_name': ['Model_A', 'Model_A', 'Model_B', 'Model_B'],
            'dataset': ['test', 'test', 'test', 'test'],
            'accuracy': [0.85, 0.87, 0.92, 0.93]
        })
        
        best = get_best_model(mock_df, 'accuracy', 'test')
        
        assert best is not None
        assert best['model'] == 'Model_B'
        assert best['score'] > 0.90
    
    def test_get_best_model_empty_dataset(self):
        """Test handling when dataset has no data"""
        mock_df = pd.DataFrame({
            'model_full_name': ['Model_A'],
            'dataset': ['train'],
            'accuracy': [0.85]
        })
        
        best = get_best_model(mock_df, 'accuracy', 'test')
        
        assert best is None


class TestPerformanceDegradation:
    """Test performance degradation detection"""
    
    def test_detect_degradation(self):
        """Test detection of significant performance drop"""
        mock_df = pd.DataFrame({
            'model_full_name': ['Model_A'] * 10,
            'evaluation_date': pd.to_datetime([f'2024-11-{i:02d}' for i in range(1, 11)]),
            'accuracy': [0.90, 0.89, 0.88, 0.87, 0.86, 0.80, 0.79, 0.78, 0.77, 0.76]
        })
        
        degraded = detect_performance_degradation(mock_df, 'Model_A', 'accuracy')
        
        assert degraded == True, "Should detect performance degradation"
    
    def test_no_degradation(self):
        """Test when performance is stable"""
        mock_df = pd.DataFrame({
            'model_full_name': ['Model_A'] * 10,
            'evaluation_date': pd.to_datetime([f'2024-11-{i:02d}' for i in range(1, 11)]),
            'accuracy': [0.85] * 10
        })
        
        degraded = detect_performance_degradation(mock_df, 'Model_A', 'accuracy')
        
        assert degraded == False, "Should not detect degradation in stable performance"


class TestPerformanceStatus:
    """Test performance status classification"""
    
    def test_excellent_accuracy(self):
        """Test excellent performance classification"""
        status, icon = get_performance_status(0.95, 'accuracy')
        assert status == 'Excellent'
        assert icon == '🟢'
    
    def test_good_accuracy(self):
        """Test good performance classification"""
        status, icon = get_performance_status(0.85, 'accuracy')
        assert status == 'Good'
        assert icon == '🟡'
    
    def test_poor_accuracy(self):
        """Test poor performance classification"""
        status, icon = get_performance_status(0.70, 'accuracy')
        assert status == 'Needs Improvement'
        assert icon == '🔴'
    
    def test_loss_metric(self):
        """Test loss metric classification (inverse logic)"""
        status, icon = get_performance_status(0.20, 'loss')
        assert status == 'Excellent'
        assert icon == '🟢'
    
    def test_inference_time(self):
        """Test inference time classification"""
        status, icon = get_performance_status(45.0, 'inference_time_ms')
        assert status == 'Fast'
        assert icon == '🟢'


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, '-v'])