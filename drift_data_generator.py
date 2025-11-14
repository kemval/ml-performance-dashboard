"""
Generate mock feature data with simulated drift
Creates reference and production data for drift detection
"""

import json
import numpy as np
from datetime import datetime, timedelta
import random


def generate_feature_distributions():
    """
    Generate mock feature distributions for drift detection
    Simulates input features (credit scores, loan amounts, etc.)
    """
    
    # Reference period (stable baseline - 60 days ago to 30 days ago)
    reference_start = datetime.now() - timedelta(days=60)
    reference_end = datetime.now() - timedelta(days=30)
    
    # Production period (recent - last 30 days, with drift)
    prod_start = datetime.now() - timedelta(days=30)
    prod_end = datetime.now()
    
    all_data = []
    
    # Features we're monitoring
    feature_configs = {
        'credit_score': {
            'reference_mean': 680,
            'reference_std': 80,
            'drift_mean_shift': 30,  # Drift: scores increasing
            'drift_std_shift': 10
        },
        'loan_amount': {
            'reference_mean': 50000,
            'reference_std': 20000,
            'drift_mean_shift': 15000,  # Drift: larger loans
            'drift_std_shift': 5000
        },
        'debt_to_income_ratio': {
            'reference_mean': 0.35,
            'reference_std': 0.15,
            'drift_mean_shift': 0.08,  # Drift: higher ratios
            'drift_std_shift': 0.03
        },
        'employment_length_years': {
            'reference_mean': 8.5,
            'reference_std': 5.0,
            'drift_mean_shift': -2.0,  # Drift: shorter employment
            'drift_std_shift': 0.5
        },
        'num_credit_lines': {
            'reference_mean': 10,
            'reference_std': 4,
            'drift_mean_shift': 3,  # Drift: more credit lines
            'drift_std_shift': 1
        }
    }
    
    # Generate REFERENCE data (stable baseline)
    print("🔵 Generating reference data (stable baseline)...")
    current_date = reference_start
    samples_per_day = 500
    
    while current_date <= reference_end:
        for _ in range(samples_per_day):
            sample = {
                'timestamp': current_date.isoformat(),
                'date': current_date.strftime('%Y-%m-%d'),
                'period': 'reference',
                'features': {}
            }
            
            # Generate features with reference distribution
            for feature, config in feature_configs.items():
                value = np.random.normal(config['reference_mean'], config['reference_std'])
                
                # Ensure realistic bounds
                if feature == 'credit_score':
                    value = max(300, min(850, value))
                elif feature == 'debt_to_income_ratio':
                    value = max(0, min(1, value))
                elif feature == 'employment_length_years':
                    value = max(0, value)
                elif feature == 'num_credit_lines':
                    value = max(1, int(value))
                
                sample['features'][feature] = round(value, 2)
            
            # Add model prediction (no concept drift yet)
            sample['prediction'] = 1 if np.random.random() < 0.65 else 0  # 65% approval rate
            sample['prediction_probability'] = round(np.random.beta(5, 3), 3)  # Skewed toward approval
            
            all_data.append(sample)
        
        current_date += timedelta(days=1)
    
    # Generate PRODUCTION data (with drift)
    print("🔴 Generating production data (with drift)...")
    current_date = prod_start
    drift_days = (prod_end - prod_start).days
    
    while current_date <= prod_end:
        # Calculate drift progression (0 to 1 over time)
        days_elapsed = (current_date - prod_start).days
        drift_factor = days_elapsed / drift_days  # Gradual drift over time
        
        for _ in range(samples_per_day):
            sample = {
                'timestamp': current_date.isoformat(),
                'date': current_date.strftime('%Y-%m-%d'),
                'period': 'production',
                'drift_factor': round(drift_factor, 3),
                'features': {}
            }
            
            # Generate features with drifted distribution
            for feature, config in feature_configs.items():
                # Apply gradual drift
                drifted_mean = config['reference_mean'] + (config['drift_mean_shift'] * drift_factor)
                drifted_std = config['reference_std'] + (config['drift_std_shift'] * drift_factor)
                
                value = np.random.normal(drifted_mean, drifted_std)
                
                # Ensure realistic bounds
                if feature == 'credit_score':
                    value = max(300, min(850, value))
                elif feature == 'debt_to_income_ratio':
                    value = max(0, min(1, value))
                elif feature == 'employment_length_years':
                    value = max(0, value)
                elif feature == 'num_credit_lines':
                    value = max(1, int(value))
                
                sample['features'][feature] = round(value, 2)
            
            # Add concept drift: approval rate changes
            approval_rate = 0.65 - (0.15 * drift_factor)  # Drops from 65% to 50%
            sample['prediction'] = 1 if np.random.random() < approval_rate else 0
            
            # Prediction distribution shifts
            if drift_factor < 0.5:
                sample['prediction_probability'] = round(np.random.beta(5, 3), 3)
            else:
                sample['prediction_probability'] = round(np.random.beta(3, 5), 3)  # Shift toward rejection
            
            all_data.append(sample)
        
        current_date += timedelta(days=1)
    
    return all_data


def save_drift_data(data, filename='data/drift_data.json'):
    """Save drift data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    reference_count = len([d for d in data if d['period'] == 'reference'])
    production_count = len([d for d in data if d['period'] == 'production'])
    
    print(f"\n✅ Generated {len(data)} total samples")
    print(f"   📊 Reference: {reference_count:,} samples")
    print(f"   📊 Production: {production_count:,} samples")
    print(f"✅ Saved to {filename}")
    
    # Print drift summary
    print(f"\n📈 SIMULATED DRIFT PATTERNS:")
    print(f"   • Credit scores: ↑ increasing (better applicants)")
    print(f"   • Loan amounts: ↑ increasing (larger loans)")
    print(f"   • Debt-to-income: ↑ increasing (riskier)")
    print(f"   • Employment length: ↓ decreasing (less stable)")
    print(f"   • Credit lines: ↑ increasing (more accounts)")
    print(f"   • Approval rate: ↓ dropping from 65% to 50%")


if __name__ == "__main__":
    print("🔬 Generating Feature Data with Drift Simulation...\n")
    drift_data = generate_feature_distributions()
    save_drift_data(drift_data)
    print("\nDone! Run the drift detector to analyze drift patterns.")