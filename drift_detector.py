"""
Drift Detection Engine
Detects data drift, concept drift, and performance drift
Uses statistical tests: KS test, Chi-square, PSI (Population Stability Index)
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """Comprehensive drift detection for ML models"""
    
    def __init__(self, reference_data, production_data):
        """
        Initialize drift detector
        
        Args:
            reference_data: Baseline/training data (list of dicts)
            production_data: Current production data (list of dicts)
        """
        self.reference_data = pd.DataFrame(reference_data)
        self.production_data = pd.DataFrame(production_data)
        
        # Extract features into separate dataframes
        self.ref_features = pd.json_normalize(self.reference_data['features'])
        self.prod_features = pd.json_normalize(self.production_data['features'])
        
        # Thresholds for drift detection
        self.ks_threshold = 0.05  # p-value threshold
        self.psi_threshold = 0.1  # PSI threshold (0.1 = moderate drift)
        self.chi2_threshold = 0.05
    
    def kolmogorov_smirnov_test(self, feature_name):
        """
        Perform Kolmogorov-Smirnov test for continuous features
        Tests if two samples come from the same distribution
        
        Returns: (statistic, p_value, drift_detected)
        """
        ref_values = self.ref_features[feature_name].dropna()
        prod_values = self.prod_features[feature_name].dropna()
        
        statistic, p_value = ks_2samp(ref_values, prod_values)
        
        drift_detected = p_value < self.ks_threshold
        
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': round(statistic, 4),
            'p_value': round(p_value, 4),
            'threshold': self.ks_threshold,
            'drift_detected': drift_detected,
            'severity': self._get_drift_severity(p_value, self.ks_threshold)
        }
    
    def population_stability_index(self, feature_name, n_bins=10):
        """
        Calculate Population Stability Index (PSI)
        Industry standard for measuring distribution shift
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        
        Returns: PSI value and interpretation
        """
        ref_values = self.ref_features[feature_name].dropna()
        prod_values = self.prod_features[feature_name].dropna()
        
        # Create bins based on reference data
        _, bin_edges = np.histogram(ref_values, bins=n_bins)
        
        # Calculate distribution in each bin
        ref_dist, _ = np.histogram(ref_values, bins=bin_edges)
        prod_dist, _ = np.histogram(prod_values, bins=bin_edges)
        
        # Convert to percentages
        ref_dist = ref_dist / len(ref_values)
        prod_dist = prod_dist / len(prod_values)
        
        # Avoid division by zero
        ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
        prod_dist = np.where(prod_dist == 0, 0.0001, prod_dist)
        
        # Calculate PSI
        psi = np.sum((prod_dist - ref_dist) * np.log(prod_dist / ref_dist))
        
        drift_detected = psi > self.psi_threshold
        
        return {
            'test': 'Population Stability Index',
            'psi_value': round(psi, 4),
            'threshold': self.psi_threshold,
            'drift_detected': drift_detected,
            'severity': self._get_psi_severity(psi)
        }
    
    def chi_square_test(self, feature_name, n_bins=10):
        """
        Chi-square test for categorical or binned continuous features
        Tests independence of distributions
        
        Returns: (statistic, p_value, drift_detected)
        """
        ref_values = self.ref_features[feature_name].dropna()
        prod_values = self.prod_features[feature_name].dropna()
        
        # Create bins
        _, bin_edges = np.histogram(
            np.concatenate([ref_values, prod_values]), 
            bins=n_bins
        )
        
        # Get frequencies
        ref_freq, _ = np.histogram(ref_values, bins=bin_edges)
        prod_freq, _ = np.histogram(prod_values, bins=bin_edges)
        
        # Create contingency table
        contingency_table = np.array([ref_freq, prod_freq])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        drift_detected = p_value < self.chi2_threshold
        
        return {
            'test': 'Chi-Square',
            'statistic': round(chi2_stat, 4),
            'p_value': round(p_value, 4),
            'degrees_of_freedom': dof,
            'threshold': self.chi2_threshold,
            'drift_detected': drift_detected,
            'severity': self._get_drift_severity(p_value, self.chi2_threshold)
        }
    
    def detect_concept_drift(self):
        """
        Detect concept drift: change in target variable distribution
        Analyzes prediction distributions
        """
        ref_predictions = self.reference_data['prediction'].dropna()
        prod_predictions = self.production_data['prediction'].dropna()
        
        # Compare prediction distributions
        ref_mean = ref_predictions.mean()
        prod_mean = prod_predictions.mean()
        
        # Statistical test
        statistic, p_value = ks_2samp(ref_predictions, prod_predictions)
        
        drift_detected = p_value < self.ks_threshold
        
        # Calculate change percentage
        change_pct = ((prod_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
        
        return {
            'test': 'Concept Drift Detection',
            'reference_approval_rate': round(ref_mean, 4),
            'production_approval_rate': round(prod_mean, 4),
            'change_percent': round(change_pct, 2),
            'p_value': round(p_value, 4),
            'drift_detected': drift_detected,
            'severity': self._get_drift_severity(p_value, self.ks_threshold)
        }
    
    def analyze_all_features(self):
        """
        Analyze all features for drift
        Returns comprehensive drift report
        """
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'reference_period': {
                'start': self.reference_data['date'].min(),
                'end': self.reference_data['date'].max(),
                'samples': len(self.reference_data)
            },
            'production_period': {
                'start': self.production_data['date'].min(),
                'end': self.production_data['date'].max(),
                'samples': len(self.production_data)
            },
            'feature_drift': {},
            'concept_drift': None,
            'summary': {
                'total_features_analyzed': 0,
                'features_with_drift': 0,
                'drift_percentage': 0
            }
        }
        
        # Analyze each feature
        features = self.ref_features.columns
        
        for feature in features:
            feature_results = {
                'feature_name': feature,
                'tests': {}
            }
            
            # Run KS test
            ks_result = self.kolmogorov_smirnov_test(feature)
            feature_results['tests']['ks_test'] = ks_result
            
            # Run PSI
            psi_result = self.population_stability_index(feature)
            feature_results['tests']['psi'] = psi_result
            
            # Run Chi-square
            chi2_result = self.chi_square_test(feature)
            feature_results['tests']['chi_square'] = chi2_result
            
            # Overall drift assessment
            drift_count = sum([
                ks_result['drift_detected'],
                psi_result['drift_detected'],
                chi2_result['drift_detected']
            ])
            
            feature_results['overall_drift_detected'] = drift_count >= 2  # Majority vote
            feature_results['drift_confidence'] = drift_count / 3  # 0 to 1
            
            # Calculate distributional statistics
            feature_results['statistics'] = {
                'reference_mean': round(self.ref_features[feature].mean(), 4),
                'production_mean': round(self.prod_features[feature].mean(), 4),
                'reference_std': round(self.ref_features[feature].std(), 4),
                'production_std': round(self.prod_features[feature].std(), 4),
                'mean_shift_percent': round(
                    ((self.prod_features[feature].mean() - self.ref_features[feature].mean()) / 
                     self.ref_features[feature].mean()) * 100, 2
                ) if self.ref_features[feature].mean() != 0 else 0
            }
            
            results['feature_drift'][feature] = feature_results
        
        # Analyze concept drift
        results['concept_drift'] = self.detect_concept_drift()
        
        # Summary statistics
        results['summary']['total_features_analyzed'] = len(features)
        results['summary']['features_with_drift'] = sum(
            1 for f in results['feature_drift'].values() if f['overall_drift_detected']
        )
        results['summary']['drift_percentage'] = round(
            (results['summary']['features_with_drift'] / results['summary']['total_features_analyzed']) * 100, 2
        ) if results['summary']['total_features_analyzed'] > 0 else 0
        
        return results
    
    def _get_drift_severity(self, p_value, threshold):
        """Categorize drift severity based on p-value"""
        if p_value >= threshold:
            return 'none'
        elif p_value >= threshold / 2:
            return 'low'
        elif p_value >= threshold / 10:
            return 'moderate'
        else:
            return 'high'
    
    def _get_psi_severity(self, psi):
        """Categorize drift severity based on PSI"""
        if psi < 0.1:
            return 'none'
        elif psi < 0.2:
            return 'moderate'
        else:
            return 'high'
    
    def generate_alerts(self, results):
        """Generate actionable alerts from drift analysis"""
        alerts = []
        
        # Feature drift alerts
        for feature, data in results['feature_drift'].items():
            if data['overall_drift_detected']:
                severity = max(
                    [data['tests']['ks_test']['severity'],
                     data['tests']['psi']['severity'],
                     data['tests']['chi_square']['severity']],
                    key=lambda x: ['none', 'low', 'moderate', 'high'].index(x)
                )
                
                alerts.append({
                    'type': 'feature_drift',
                    'severity': severity,
                    'feature': feature,
                    'message': f"Data drift detected in {feature}",
                    'mean_shift': data['statistics']['mean_shift_percent'],
                    'recommendation': self._get_recommendation(severity)
                })
        
        # Concept drift alert
        if results['concept_drift']['drift_detected']:
            severity = results['concept_drift']['severity']
            alerts.append({
                'type': 'concept_drift',
                'severity': severity,
                'message': f"Concept drift detected: Approval rate changed by {results['concept_drift']['change_percent']:.1f}%",
                'recommendation': "Model retraining recommended"
            })
        
        return alerts
    
    def _get_recommendation(self, severity):
        """Get recommendation based on drift severity"""
        recommendations = {
            'none': 'Continue monitoring',
            'low': 'Monitor closely, no immediate action needed',
            'moderate': 'Investigate root cause, consider retraining',
            'high': 'Urgent: Retrain model or adjust data pipeline'
        }
        return recommendations.get(severity, 'Investigate further')


def load_drift_data(filepath='data/drift_data.json'):
    """Load drift data from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_drift_report(results, alerts, filename='data/drift_report.json'):
    """Save drift analysis report"""
    
    # Helper function to convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_types(obj.tolist())
        else:
            return obj
    
    report = {
        'drift_analysis': convert_types(results),
        'alerts': convert_types(alerts),
        'generated_at': datetime.now().isoformat()
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Drift report saved to {filename}")


if __name__ == "__main__":
    print("🔬 Running Drift Detection Analysis...\n")
    
    # Load data
    print("📊 Loading data...")
    data = load_drift_data()
    
    reference = [d for d in data if d['period'] == 'reference']
    production = [d for d in data if d['period'] == 'production']
    
    print(f"   Reference: {len(reference):,} samples")
    print(f"   Production: {len(production):,} samples\n")
    
    # Initialize detector
    print("🔍 Analyzing drift patterns...")
    detector = DriftDetector(reference, production)
    
    # Run analysis
    results = detector.analyze_all_features()
    alerts = detector.generate_alerts(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 DRIFT DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total Features Analyzed: {results['summary']['total_features_analyzed']}")
    print(f"Features with Drift: {results['summary']['features_with_drift']}")
    print(f"Drift Percentage: {results['summary']['drift_percentage']:.1f}%")
    print()
    
    print("🚨 ALERTS:")
    if len(alerts) == 0:
        print("   ✅ No significant drift detected")
    else:
        for alert in alerts:
            severity_icon = {'low': '🟡', 'moderate': '🟠', 'high': '🔴'}.get(alert['severity'], '⚪')
            print(f"   {severity_icon} [{alert['severity'].upper()}] {alert['message']}")
            print(f"      → {alert['recommendation']}")
    
    print("=" * 70)
    
    # Save report
    save_drift_report(results, alerts)
    print("\nAnalysis complete! Check drift_report.json for details.")