import json
import pandas as pd

def load_data_from_json(filepath="data/model_results.json"):
    """
    Load model evaluation data from JSON file
    In production, this would connect to AWS S3 using boto3:
    
    import boto3
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='moody-ml-results', Key='model_results.json')
    data = json.loads(obj['Body'].read())
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Run 'python data_generator.py' first!")
        return []

def prepare_dataframe(data):
    """Convert JSON data to pandas DataFrame with proper types"""
    if not data:
        return pd.DataFrame()
    
    # Flatten nested metrics
    records = []
    for item in data:
        record = {
            'model_name': item['model_name'],
            'model_version': item['model_version'],
            'model_full_name': item['model_full_name'],
            'evaluation_date': item['evaluation_date'],
            'dataset': item['dataset'],
            'accuracy': item['metrics']['accuracy'],
            'precision': item['metrics']['precision'],
            'recall': item['metrics']['recall'],
            'f1_score': item['metrics']['f1_score'],
            'inference_time_ms': item['metrics']['inference_time_ms'],
            'loss': item['metrics']['loss'],
            'samples_evaluated': item['metadata']['samples_evaluated'],
            'aws_s3_path': item['metadata']['aws_s3_path']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df['evaluation_date'] = pd.to_datetime(df['evaluation_date'])
    
    return df

def calculate_performance_change(df, model_name, metric='accuracy'):
    """Calculate percentage change in performance over time"""
    model_data = df[df['model_full_name'] == model_name].sort_values('evaluation_date')
    
    if len(model_data) < 2:
        return 0
    
    first_value = model_data[metric].iloc[0]
    last_value = model_data[metric].iloc[-1]
    
    if first_value == 0:
        return 0
    
    change = ((last_value - first_value) / first_value) * 100
    return round(change, 2)

def get_best_model(df, metric='accuracy', dataset='test'):
    """Identify best performing model for a given metric and dataset"""
    filtered = df[df['dataset'] == dataset]
    
    if filtered.empty:
        return None
    
    # Get average performance per model
    avg_performance = filtered.groupby('model_full_name')[metric].mean()
    best_model = avg_performance.idxmax()
    best_score = avg_performance.max()
    
    return {
        'model': best_model,
        'score': round(best_score, 4)
    }

def detect_performance_degradation(df, model_name, metric='accuracy', threshold=-0.05):
    """
    Detect if model performance has degraded significantly
    Returns True if recent performance dropped by more than threshold
    """
    model_data = df[df['model_full_name'] == model_name].sort_values('evaluation_date')
    
    if len(model_data) < 5:
        return False
    
    # Compare recent average (last 5) vs earlier average (first 5)
    recent_avg = model_data[metric].tail(5).mean()
    earlier_avg = model_data[metric].head(5).mean()
    
    change = (recent_avg - earlier_avg) / earlier_avg
    
    return change < threshold

def get_performance_status(value, metric='accuracy'):
    """
    Return color-coded status based on performance thresholds
    Returns: ('status', 'color')
    """
    if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if value >= 0.90:
            return ('Excellent', '🟢')
        elif value >= 0.80:
            return ('Good', '🟡')
        else:
            return ('Needs Improvement', '🔴')
    
    elif metric == 'loss':
        if value <= 0.25:
            return ('Excellent', '🟢')
        elif value <= 0.40:
            return ('Good', '🟡')
        else:
            return ('Needs Improvement', '🔴')
    
    elif metric == 'inference_time_ms':
        if value <= 50:
            return ('Fast', '🟢')
        elif value <= 100:
            return ('Moderate', '🟡')
        else:
            return ('Slow', '🔴')
    
    return ('Unknown', '⚪')

def export_to_csv(df, filename="model_performance_export.csv"):
    """Export filtered data to CSV"""
    df.to_csv(filename, index=False)
    return filename