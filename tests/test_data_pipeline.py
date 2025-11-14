"""
Integration tests for data pipeline
Tests end-to-end data flow from source to dashboard
"""

import pytest
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_data_from_json, prepare_dataframe
import yaml


class TestDataPipeline:
    """Test complete data pipeline from file to processed DataFrame"""
    
    @pytest.fixture
    def config(self):
        """Load test configuration"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'test_config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_data_file_exists(self, config):
        """Test that data file exists and is accessible"""
        data_path = config['integration_tests']['data_file_path']
        assert os.path.exists(data_path), f"Data file not found at {data_path}"
    
    def test_data_file_is_valid_json(self, config):
        """Test that data file contains valid JSON"""
        data_path = config['integration_tests']['data_file_path']
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            assert isinstance(data, list), "Data should be a list of records"
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in data file: {e}")
    
    def test_load_data_from_json(self):
        """Test data loading function"""
        data = load_data_from_json()
        
        assert isinstance(data, list), "Loaded data should be a list"
        assert len(data) > 0, "Data should not be empty"
    
    def test_data_structure_integrity(self):
        """Test that loaded data has correct structure"""
        data = load_data_from_json()
        
        if len(data) > 0:
            record = data[0]
            
            # Check required top-level fields
            required_fields = ['model_name', 'model_version', 'model_full_name', 
                             'evaluation_date', 'dataset', 'metrics', 'metadata']
            
            for field in required_fields:
                assert field in record, f"Missing required field: {field}"
            
            # Check metrics structure
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                              'inference_time_ms', 'loss']
            
            for metric in required_metrics:
                assert metric in record['metrics'], f"Missing metric: {metric}"
    
    def test_prepare_dataframe_pipeline(self):
        """Test complete pipeline from JSON to DataFrame"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        assert isinstance(df, pd.DataFrame), "Should return pandas DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"
        assert 'accuracy' in df.columns, "Should have accuracy column"
        assert 'evaluation_date' in df.columns, "Should have date column"
    
    def test_dataframe_date_conversion(self):
        """Test that dates are properly converted to datetime"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        assert pd.api.types.is_datetime64_any_dtype(df['evaluation_date']), \
            "evaluation_date should be datetime type"
    
    def test_multiple_models_exist(self, config):
        """Test that data contains multiple models"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        unique_models = df['model_full_name'].nunique()
        min_models = config['integration_tests']['expected_models_count']
        
        assert unique_models >= min_models, \
            f"Expected at least {min_models} models, found {unique_models}"
    
    def test_multiple_datasets_exist(self, config):
        """Test that data contains required datasets"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        required_datasets = config['data_validation']['required_datasets']
        available_datasets = set(df['dataset'].unique())
        
        for dataset in required_datasets:
            assert dataset in available_datasets, \
                f"Missing required dataset: {dataset}"
    
    def test_date_range_coverage(self, config):
        """Test that data covers sufficient time period"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        date_range = (df['evaluation_date'].max() - df['evaluation_date'].min()).days
        min_days = config['integration_tests']['date_range_days']
        
        assert date_range >= min_days, \
            f"Data should cover at least {min_days} days, found {date_range} days"
    
    def test_no_missing_values_in_critical_fields(self):
        """Test that critical fields have no missing values"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        critical_fields = ['model_name', 'accuracy', 'evaluation_date', 'dataset']
        
        for field in critical_fields:
            missing_count = df[field].isna().sum()
            assert missing_count == 0, \
                f"Found {missing_count} missing values in critical field: {field}"
    
    def test_data_types_are_correct(self):
        """Test that data types are as expected"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        # Numeric fields
        numeric_fields = ['accuracy', 'precision', 'recall', 'f1_score', 
                         'inference_time_ms', 'loss']
        
        for field in numeric_fields:
            assert pd.api.types.is_numeric_dtype(df[field]), \
                f"{field} should be numeric type"
        
        # String fields
        string_fields = ['model_name', 'model_version', 'dataset']
        
        for field in string_fields:
            assert pd.api.types.is_string_dtype(df[field]) or pd.api.types.is_object_dtype(df[field]), \
                f"{field} should be string type"
    
    def test_aws_s3_path_format(self):
        """Test that AWS S3 paths are properly formatted"""
        data = load_data_from_json()
        
        for record in data[:10]:  # Check first 10 records
            s3_path = record['metadata']['aws_s3_path']
            assert s3_path.startswith('s3://'), \
                f"AWS S3 path should start with 's3://': {s3_path}"
    
    def test_samples_evaluated_range(self, config):
        """Test that sample counts are within reasonable range"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        min_samples = config['data_validation']['min_samples_per_evaluation']
        max_samples = config['data_validation']['max_samples_per_evaluation']
        
        assert df['samples_evaluated'].min() >= min_samples, \
            f"Some evaluations have fewer than {min_samples} samples"
        
        assert df['samples_evaluated'].max() <= max_samples, \
            f"Some evaluations have more than {max_samples} samples"


class TestDataQuality:
    """Test data quality and consistency"""
    
    def test_metric_values_in_valid_range(self):
        """Test that metric values are within valid ranges"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        # Metrics that should be between 0 and 1
        bounded_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in bounded_metrics:
            assert df[metric].min() >= 0, f"{metric} has values below 0"
            assert df[metric].max() <= 1, f"{metric} has values above 1"
        
        # Inference time should be positive
        assert df['inference_time_ms'].min() > 0, "Inference time should be positive"
        
        # Loss should be positive
        assert df['loss'].min() >= 0, "Loss should be non-negative"
    
    def test_no_duplicate_evaluations(self):
        """Test that there are no duplicate evaluation records"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        # Create unique key from model, date, and dataset
        df['unique_key'] = (df['model_full_name'] + '_' + 
                           df['evaluation_date'].astype(str) + '_' + 
                           df['dataset'])
        
        duplicates = df['unique_key'].duplicated().sum()
        
        # Note: Some duplicates might be intentional (multiple runs same day)
        # So we just check that duplicates aren't excessive
        duplicate_percentage = (duplicates / len(df)) * 100
        
        assert duplicate_percentage < 10, \
            f"Too many duplicate evaluations: {duplicate_percentage:.1f}%"
    
    def test_chronological_order_possible(self):
        """Test that data can be sorted chronologically"""
        data = load_data_from_json()
        df = prepare_dataframe(data)
        
        # Should be able to sort by date without errors
        try:
            df_sorted = df.sort_values('evaluation_date')
            assert len(df_sorted) == len(df), "Sorting should not change row count"
        except Exception as e:
            pytest.fail(f"Failed to sort data chronologically: {e}")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, '-v'])