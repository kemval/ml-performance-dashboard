import json
import random
from datetime import datetime, timedelta
import numpy as np

def generate_model_results():
    """
    Generate mock ML model evaluation data for dashboard
    Simulates realistic model performance metrics over time
    """
    
    models = [
        {"name": "CreditRiskModel_A", "version": "v1.0"},
        {"name": "CreditRiskModel_A", "version": "v1.1"},
        {"name": "CreditRiskModel_A", "version": "v2.0"},
        {"name": "FraudDetectionModel_B", "version": "v1.0"},
        {"name": "FraudDetectionModel_B", "version": "v1.5"},
        {"name": "DefaultPredictionModel_C", "version": "v1.0"},
    ]
    
    datasets = ["train", "validation", "test", "production"]
    
    results = []
    
    # Generate data for last 60 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    for model in models:
        # Base performance varies by model
        if "CreditRisk" in model["name"]:
            base_accuracy = 0.85 if model["version"] == "v1.0" else 0.88 if model["version"] == "v1.1" else 0.91
            base_precision = 0.83 if model["version"] == "v1.0" else 0.86 if model["version"] == "v1.1" else 0.89
        elif "FraudDetection" in model["name"]:
            base_accuracy = 0.92 if model["version"] == "v1.0" else 0.94
            base_precision = 0.90 if model["version"] == "v1.0" else 0.93
        else:
            base_accuracy = 0.87
            base_precision = 0.85
        
        # Generate evaluations over time
        current_date = start_date
        eval_count = 0
        
        while current_date <= end_date:
            # More frequent evaluations for newer versions
            days_increment = 2 if model["version"] in ["v2.0", "v1.5"] else 4
            
            for dataset in datasets:
                # Add some variance and trends
                noise = np.random.normal(0, 0.02)
                trend = 0.01 if model["version"] in ["v2.0", "v1.5"] else 0
                
                # Production typically performs slightly worse
                dataset_penalty = -0.03 if dataset == "production" else 0
                
                accuracy = max(0.7, min(0.99, base_accuracy + noise + trend + dataset_penalty))
                precision = max(0.7, min(0.99, base_precision + noise + trend + dataset_penalty))
                recall = max(0.7, min(0.99, accuracy - 0.02 + np.random.normal(0, 0.015)))
                f1_score = 2 * (precision * recall) / (precision + recall)
                
                # Inference time (ms) - newer models might be slightly slower but more accurate
                base_inference = 45 if model["version"] == "v1.0" else 52 if model["version"] in ["v1.1", "v1.5"] else 58
                inference_time = max(20, base_inference + np.random.normal(0, 8))
                
                # Loss decreases over time and with better versions
                base_loss = 0.35 if model["version"] == "v1.0" else 0.28 if model["version"] in ["v1.1", "v1.5"] else 0.22
                loss = max(0.1, base_loss + np.random.normal(0, 0.04))
                
                result = {
                    "model_name": model["name"],
                    "model_version": model["version"],
                    "model_full_name": f"{model['name']}_{model['version']}",
                    "evaluation_date": current_date.strftime("%Y-%m-%d"),
                    "evaluation_timestamp": current_date.isoformat(),
                    "dataset": dataset,
                    "metrics": {
                        "accuracy": round(accuracy, 4),
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1_score": round(f1_score, 4),
                        "inference_time_ms": round(inference_time, 2),
                        "loss": round(loss, 4)
                    },
                    "metadata": {
                        "samples_evaluated": random.randint(5000, 50000),
                        "evaluation_id": f"eval_{eval_count}_{dataset}",
                        "aws_s3_path": f"s3://moody-ml-results/{model['name']}/{model['version']}/{current_date.strftime('%Y-%m-%d')}/{dataset}.json"
                    }
                }
                
                results.append(result)
            
            current_date += timedelta(days=days_increment)
            eval_count += 1
    
    return results

def save_results_to_json(results, filename="data/model_results.json"):
    """Save generated results to JSON file (simulates S3 storage)"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Generated {len(results)} evaluation records")
    print(f"Saved to {filename}")
    print(f"\nSummary:")
    print(f"   - Models: {len(set([r['model_full_name'] for r in results]))}")
    print(f"   - Date range: {min([r['evaluation_date'] for r in results])} to {max([r['evaluation_date'] for r in results])}")
    print(f"   - Datasets: {', '.join(set([r['dataset'] for r in results]))}")

if __name__ == "__main__":
    print("Generating mock ML model evaluation data...\n")
    results = generate_model_results()
    save_results_to_json(results)
    print("\nDone! Run 'streamlit run app.py' to see your dashboard!")