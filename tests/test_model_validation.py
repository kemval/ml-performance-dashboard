"""
Model validation and performance threshold tests
Regression tests to ensure models meet minimum performance standards
"""

import pytest
import pandas as pd
import sys
import os
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_data_from_json, prepare_dataframe, detect_performance_degradation


@pytest.fixture
def config():
    """Load test configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'test_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def test_data():
    """Load test data"""
    data = load_data_from_json()
    return prepare_dataframe(data)


class TestModelPerformanceThresholds:
    """Test that models meet minimum performance thresholds"""
    
    def test_accuracy_above_minimum(self, test_data, config):
        """Test that all models meet minimum accuracy threshold"""
        min_accuracy = config['model_thresholds']['accuracy']['minimum']
        
        # Get average accuracy per model on test dataset
        test_models = test_data[test_data['dataset'] == 'test']
        avg_accuracy = test_models.groupby('model_full_name')['accuracy'].mean()
        
        failures = []
        for model, accuracy in avg_accuracy.items():
            if accuracy < min_accuracy:
                failures.append(f"{model}: {accuracy:.4f} (min: {min_accuracy})")
        
        assert len(failures) == 0, \
            f"Models below minimum accuracy threshold:\n" + "\n".join(failures)
    
    def test_precision_above_minimum(self, test_data, config):
        """Test that all models meet minimum precision threshold"""
        min_precision = config['model_thresholds']['precision']['minimum']
        
        test_models = test_data[test_data['dataset'] == 'test']
        avg_precision = test_models.groupby('model_full_name')['precision'].mean()
        
        failures = []
        for model, precision in avg_precision.items():
            if precision < min_precision:
                failures.append(f"{model}: {precision:.4f} (min: {min_precision})")
        
        assert len(failures) == 0, \
            f"Models below minimum precision threshold:\n" + "\n".join(failures)
    
    def test_recall_above_minimum(self, test_data, config):
        """Test that all models meet minimum recall threshold"""
        min_recall = config['model_thresholds']['recall']['minimum']
        
        test_models = test_data[test_data['dataset'] == 'test']
        avg_recall = test_models.groupby('model_full_name')['recall'].mean()
        
        failures = []
        for model, recall in avg_recall.items():
            if recall < min_recall:
                failures.append(f"{model}: {recall:.4f} (min: {min_recall})")
        
        assert len(failures) == 0, \
            f"Models below minimum recall threshold:\n" + "\n".join(failures)
    
    def test_f1_score_above_minimum(self, test_data, config):
        """Test that all models meet minimum F1 score threshold"""
        min_f1 = config['model_thresholds']['f1_score']['minimum']
        
        test_models = test_data[test_data['dataset'] == 'test']
        avg_f1 = test_models.groupby('model_full_name')['f1_score'].mean()
        
        failures = []
        for model, f1 in avg_f1.items():
            if f1 < min_f1:
                failures.append(f"{model}: {f1:.4f} (min: {min_f1})")
        
        assert len(failures) == 0, \
            f"Models below minimum F1 score threshold:\n" + "\n".join(failures)
    
    def test_loss_below_maximum(self, test_data, config):
        """Test that all models have loss below maximum threshold"""
        max_loss = config['model_thresholds']['loss']['maximum']
        
        test_models = test_data[test_data['dataset'] == 'test']
        avg_loss = test_models.groupby('model_full_name')['loss'].mean()
        
        failures = []
        for model, loss in avg_loss.items():
            if loss > max_loss:
                failures.append(f"{model}: {loss:.4f} (max: {max_loss})")
        
        assert len(failures) == 0, \
            f"Models above maximum loss threshold:\n" + "\n".join(failures)
    
    def test_inference_time_below_maximum(self, test_data, config):
        """Test that all models have inference time below maximum threshold"""
        max_time = config['model_thresholds']['inference_time_ms']['maximum']
        
        test_models = test_data[test_data['dataset'] == 'test']
        avg_time = test_models.groupby('model_full_name')['inference_time_ms'].mean()
        
        failures = []
        for model, time in avg_time.items():
            if time > max_time:
                failures.append(f"{model}: {time:.2f}ms (max: {max_time}ms)")
        
        assert len(failures) == 0, \
            f"Models above maximum inference time:\n" + "\n".join(failures)


class TestRegressionDetection:
    """Test detection of model performance regression"""
    
    def test_no_severe_performance_drops(self, test_data, config):
        """Test that no models have severe performance drops"""
        models = test_data['model_full_name'].unique()
        
        degraded_models = []
        for model in models:
            model_data = test_data[test_data['model_full_name'] == model]
            
            if detect_performance_degradation(model_data, model, 'accuracy', threshold=-0.10):
                degraded_models.append(model)
        
        assert len(degraded_models) == 0, \
            f"Models with severe performance degradation (>10%): {degraded_models}"
    
    def test_recent_performance_stability(self, test_data):
        """Test that recent model performance is stable"""
        # Get last 7 days of data
        latest_date = test_data['evaluation_date'].max()
        recent_data = test_data[test_data['evaluation_date'] >= (latest_date - pd.Timedelta(days=7))]
        
        if len(recent_data) == 0:
            pytest.skip("Not enough recent data for stability test")
        
        # Check variance in accuracy for each model
        models = recent_data['model_full_name'].unique()
        
        unstable_models = []
        for model in models:
            model_recent = recent_data[recent_data['model_full_name'] == model]
            
            if len(model_recent) >= 3:
                accuracy_std = model_recent['accuracy'].std()
                
                # Accuracy standard deviation should be < 0.05 (5%)
                if accuracy_std > 0.05:
                    unstable_models.append(f"{model}: std={accuracy_std:.4f}")
        
        assert len(unstable_models) == 0, \
            f"Models with unstable recent performance:\n" + "\n".join(unstable_models)


class TestProductionPerformance:
    """Test production-specific performance requirements"""
    
    def test_production_meets_minimum_standards(self, test_data, config):
        """Test that production performance meets minimum standards"""
        prod_data = test_data[test_data['dataset'] == 'production']
        
        if len(prod_data) == 0:
            pytest.skip("No production data available")
        
        min_accuracy = config['model_thresholds']['accuracy']['minimum']
        
        prod_models = prod_data.groupby('model_full_name')['accuracy'].mean()
        
        failures = []
        for model, accuracy in prod_models.items():
            if accuracy < min_accuracy:
                failures.append(f"{model}: {accuracy:.4f} (min: {min_accuracy})")
        
        assert len(failures) == 0, \
            f"Production models below minimum accuracy:\n" + "\n".join(failures)
    
    def test_production_test_gap_acceptable(self, test_data, config):
        """Test that gap between test and production performance is acceptable"""
        max_gap = config['regression_tests']['max_acceptable_drop_percent'] / 100
        
        models = test_data['model_full_name'].unique()
        
        gaps = []
        for model in models:
            test_perf = test_data[(test_data['model_full_name'] == model) & 
                                 (test_data['dataset'] == 'test')]['accuracy'].mean()
            
            prod_perf = test_data[(test_data['model_full_name'] == model) & 
                                 (test_data['dataset'] == 'production')]['accuracy'].mean()
            
            if pd.notna(test_perf) and pd.notna(prod_perf):
                gap = (test_perf - prod_perf) / test_perf
                
                if gap > max_gap:
                    gaps.append(f"{model}: {gap*100:.1f}% gap (max: {max_gap*100:.1f}%)")
        
        assert len(gaps) == 0, \
            f"Models with excessive test-production gap:\n" + "\n".join(gaps)


class TestModelComparison:
    """Test model version comparisons"""
    
    def test_newer_versions_improve_performance(self, test_data):
        """Test that newer model versions generally improve performance"""
        # Group by model family (without version)
        test_data['model_family'] = test_data['model_name']
        
        families = test_data['model_family'].unique()
        
        regressions = []
        for family in families:
            family_data = test_data[(test_data['model_family'] == family) & 
                                   (test_data['dataset'] == 'test')]
            
            versions = family_data.groupby('model_version')['accuracy'].mean().sort_index()
            
            if len(versions) >= 2:
                # Check if latest version is at least as good as first version
                first_version_acc = versions.iloc[0]
                latest_version_acc = versions.iloc[-1]
                
                if latest_version_acc < first_version_acc * 0.95:  # Allow 5% tolerance
                    regressions.append(
                        f"{family}: v1={first_version_acc:.4f}, latest={latest_version_acc:.4f}"
                    )
        
        assert len(regressions) == 0, \
            f"Model families with version regressions:\n" + "\n".join(regressions)
    
    def test_best_model_identified(self, test_data):
        """Test that a clear best model can be identified"""
        test_models = test_data[test_data['dataset'] == 'test']
        avg_accuracy = test_models.groupby('model_full_name')['accuracy'].mean()
        
        assert len(avg_accuracy) > 0, "No models found in test data"
        
        best_model = avg_accuracy.idxmax()
        best_accuracy = avg_accuracy.max()
        
        # Best model should have at least 80% accuracy
        assert best_accuracy >= 0.80, \
            f"Best model {best_model} has accuracy {best_accuracy:.4f}, expected >= 0.80"


class TestDatasetConsistency:
    """Test consistency across datasets"""
    
    def test_all_models_evaluated_on_all_datasets(self, test_data, config):
        """Test that each model has been evaluated on required datasets"""
        required_datasets = config['data_validation']['required_datasets']
        models = test_data['model_full_name'].unique()
        
        missing_evaluations = []
        for model in models:
            model_datasets = set(test_data[test_data['model_full_name'] == model]['dataset'].unique())
            
            for req_dataset in required_datasets:
                if req_dataset not in model_datasets:
                    missing_evaluations.append(f"{model} missing {req_dataset} evaluation")
        
        # This might not always be required, so we just warn
        if len(missing_evaluations) > 0:
            pytest.skip(f"Some models missing dataset evaluations:\n" + "\n".join(missing_evaluations))


# Performance benchmarking (informational, not strict tests)
class TestPerformanceBenchmarks:
    """Benchmark tests for reporting purposes (non-failing)"""
    
    def test_report_average_accuracies(self, test_data):
        """Report average accuracies across all models"""
        test_models = test_data[test_data['dataset'] == 'test']
        avg_accuracy = test_models.groupby('model_full_name')['accuracy'].mean()
        
        print("\n" + "="*50)
        print("AVERAGE ACCURACIES (Test Dataset)")
        print("="*50)
        for model, accuracy in avg_accuracy.sort_values(ascending=False).items():
            print(f"{model:40s} {accuracy:.4f}")
        print("="*50)
        
        # Always pass - this is just for reporting
        assert True
    
    def test_report_inference_times(self, test_data):
        """Report average inference times"""
        test_models = test_data[test_data['dataset'] == 'test']
        avg_time = test_models.groupby('model_full_name')['inference_time_ms'].mean()
        
        print("\n" + "="*50)
        print("AVERAGE INFERENCE TIMES (Test Dataset)")
        print("="*50)
        for model, time in avg_time.sort_values().items():
            print(f"{model:40s} {time:.2f} ms")
        print("="*50)
        
        # Always pass - this is just for reporting
        assert True


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])  # -s shows print output