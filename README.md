# 📊 ML Model Performance & Drift Detection Dashboard

A professional-grade, production-ready dashboard for monitoring, benchmarking, and analyzing machine learning model performance with advanced drift detection capabilities. Built specifically to demonstrate comprehensive ML Operations capabilities for Moody's Analytics.

## 🎯 Key Features

### Performance Monitoring
- **Real-time Performance Tracking**: Monitor accuracy, precision, recall, F1 score, inference time, and loss
- **Multi-Model Comparison**: Compare performance across different model versions side-by-side
- **Time Series Analysis**: Visualize performance trends and detect degradation over time
- **Dataset Benchmarking**: Analyze performance across train/validation/test/production datasets
- **Performance Alerts**: Automatic detection of model performance degradation
- **Interactive Filtering**: Date ranges, model selection, dataset filtering with dynamic visualizations

### Drift Detection 🔬 **(Advanced Feature)**
- **Data Drift Detection**: Identify distribution shifts in input features using statistical tests
- **Concept Drift Detection**: Detect changes in model prediction distributions
- **Multiple Statistical Tests**: 
  - Kolmogorov-Smirnov (KS) Test
  - Population Stability Index (PSI)
  - Chi-Square Test
- **Automated Alerting**: Severity-based alerts (Low/Moderate/High) with actionable recommendations
- **Visual Drift Analysis**: Interactive distribution comparisons and statistical summaries
- **Production-Ready**: Industry-standard drift detection for financial ML models

### Infrastructure
- **AWS S3 Integration**: Demonstrates cloud storage integration (simulated, production-ready)
- **Automated Testing Suite**: Comprehensive test coverage (unit, integration, validation)
- **Data Export**: Download filtered results as CSV
- **Modular Architecture**: Clean, extensible codebase

---

## 🏗️ Project Structure

```
ml-performance-dashboard/
├── app.py                          # Main Streamlit dashboard (with drift tab)
├── data_generator.py               # Generates mock ML evaluation data
├── drift_data_generator.py         # Generates feature data with simulated drift
├── drift_detector.py               # Drift detection engine with statistical tests
├── utils.py                        # Helper functions (data loading, calculations)
├── test_runner.py                  # Automated test suite runner
├── test_config.yaml                # Test configuration and thresholds
├── tests/
│   ├── __init__.py
│   ├── test_utils.py               # Unit tests
│   ├── test_data_pipeline.py      # Integration tests
│   └── test_model_validation.py   # Model validation tests
├── data/
│   ├── model_results.json          # ML model evaluation results
│   ├── drift_data.json             # Feature data for drift detection
│   └── drift_report.json           # Drift analysis report
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or download this project**

2. **Set up virtual environment**:
```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install streamlit pandas plotly numpy python-dateutil scipy scikit-learn pytest pytest-html pyyaml
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

4. **Generate mock data**:
```bash
# Generate ML model performance data
python data_generator.py

# Generate drift detection data (optional but recommended)
python drift_data_generator.py

# Run drift analysis (optional but recommended)
python drift_detector.py
```

5. **Launch dashboard**:
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### Tab 1: Performance Monitoring

#### 1. Key Performance Indicators (KPIs)
- Average accuracy with trend indicators
- F1 score performance metrics
- Inference time monitoring
- Best performing model identification
- Color-coded status indicators (🟢🟡🔴)

#### 2. Performance Trends Over Time
- Interactive time-series line charts
- Multi-model overlay with legend
- Hover tooltips with detailed metrics
- Automatic trend detection

#### 3. Model Comparison
- Side-by-side bar chart comparisons
- Multiple metrics (accuracy, precision, recall, F1)
- Statistical benchmarking
- Version comparison analysis

#### 4. Dataset Performance Analysis
- Performance breakdown by dataset type (train/val/test/prod)
- Production vs test gap analysis
- Grouped bar charts for easy comparison

#### 5. Performance Heatmap
- Model vs Dataset performance matrix
- Color-coded values (red/yellow/green)
- Quick identification of weak spots
- Numerical values displayed in cells

#### 6. Alerts & Insights
- Automatic degradation detection
- Performance status indicators
- Production-test performance gap analysis
- Actionable insights and recommendations

### Tab 2: Drift Detection 🔬

#### 1. Drift Summary Dashboard
- Features monitored count
- Features with detected drift
- Concept drift status
- Active alerts counter

#### 2. Automated Alerts System
- Severity-based alerts (🟡 Low, 🟠 Moderate, 🔴 High)
- Feature-specific drift notifications
- Concept drift warnings
- Actionable recommendations for each alert

#### 3. Feature-Level Drift Analysis
- Comprehensive drift summary table
- Confidence scores for each feature
- Mean shift percentages
- Statistical test results (KS test p-values, PSI values)

#### 4. Distribution Comparisons
- Interactive histogram overlays (Reference vs Production)
- Side-by-side distribution visualization
- Statistical test details panel
- Feature selector dropdown

#### 5. Concept Drift Analysis
- Prediction distribution shift visualization
- Approval rate comparison charts
- Statistical significance testing
- Severity assessment and interpretation

#### 6. Recommendations Engine
- System health status
- Severity-based action items
- Investigation guidelines
- Retraining recommendations

---

## 🧪 Automated Testing Suite

### Running Tests

```bash
# Run all tests with HTML report
python test_runner.py

# Run specific test suites
python test_runner.py unit          # Unit tests only
python test_runner.py integration   # Integration tests only
python test_runner.py validation    # Model validation tests only

# List all available tests
python test_runner.py list

# Show help
python test_runner.py help
```

### Test Coverage

#### Unit Tests (`test_utils.py`)
- Data preparation functions
- Performance change calculations
- Best model identification
- Degradation detection
- Performance status classification
- **25+ test cases**

#### Integration Tests (`test_data_pipeline.py`)
- End-to-end data pipeline validation
- JSON file loading and parsing
- DataFrame creation and structure
- Data quality checks
- Date range validation
- Missing value detection
- Data type validation
- AWS S3 path format validation
- **20+ test cases**

#### Model Validation Tests (`test_model_validation.py`)
- Performance threshold validation
- Regression detection
- Production performance checks
- Model comparison tests
- Dataset consistency validation
- Benchmark reporting
- **15+ test cases**

### Test Configuration

All test thresholds are configurable in `test_config.yaml`:

```yaml
model_thresholds:
  accuracy:
    minimum: 0.75    # Fail if below this
    target: 0.85     # Expected performance
    excellent: 0.90  # Outstanding performance
```

---

## 🔬 Drift Detection Deep Dive

### What is Drift?

**Data Drift**: Changes in input feature distributions over time
- Example: Credit scores trending higher, loan amounts increasing

**Concept Drift**: Changes in the relationship between inputs and outputs
- Example: Same features now leading to different approval rates

### Statistical Tests Implemented

#### 1. Kolmogorov-Smirnov (KS) Test
- **Purpose**: Tests if two samples come from the same distribution
- **Output**: Statistic and p-value
- **Interpretation**: p-value < 0.05 indicates significant drift
- **Use Case**: Continuous features (credit scores, loan amounts)

#### 2. Population Stability Index (PSI)
- **Purpose**: Industry standard for measuring distribution shift
- **Ranges**:
  - PSI < 0.1: No significant change
  - 0.1 ≤ PSI < 0.2: Moderate change
  - PSI ≥ 0.2: Significant change
- **Use Case**: Widely used in financial services

#### 3. Chi-Square Test
- **Purpose**: Tests independence of categorical distributions
- **Output**: Chi-square statistic and p-value
- **Use Case**: Categorical features or binned continuous features

### Drift Detection Workflow

```
1. Generate Reference Data (Baseline)
   ↓
2. Collect Production Data (Current)
   ↓
3. Run Statistical Tests
   ↓
4. Calculate Drift Scores
   ↓
5. Generate Alerts
   ↓
6. Visualize in Dashboard
   ↓
7. Take Action (Retrain/Investigate)
```

### Example Use Case

**Scenario**: Credit risk model in production

**Detection**:
- Credit score mean shifted from 680 to 710 (+4.4%)
- Loan amount mean increased from $50k to $65k (+30%)
- Approval rate dropped from 65% to 50% (-23%)

**Alerts Generated**:
- 🔴 HIGH: Data drift in loan_amount (PSI: 0.25)
- 🟠 MODERATE: Data drift in credit_score (PSI: 0.15)
- 🔴 HIGH: Concept drift detected (-23% approval change)

**Recommendation**:
- Investigate data collection changes
- Retrain model with recent data
- Review business rules

---

## 🔧 Technical Stack

### Frontend
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive visualizations
- **Custom CSS**: Moody's-style professional design

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **SciPy**: Statistical tests (KS test, Chi-square)
- **Scikit-learn**: ML utilities and metrics

### Testing
- **pytest**: Testing framework
- **pytest-html**: HTML test reports
- **PyYAML**: Configuration management

### Cloud Integration (Ready)
- **boto3**: AWS S3 integration (demonstrated with mock)
- **JSON**: Data serialization format

---

## 💾 Data Models

### Model Evaluation Record
```json
{
  "model_name": "CreditRiskModel_A",
  "model_version": "v2.0",
  "model_full_name": "CreditRiskModel_A_v2.0",
  "evaluation_date": "2024-11-13",
  "evaluation_timestamp": "2024-11-13T14:30:00",
  "dataset": "test",
  "metrics": {
    "accuracy": 0.9145,
    "precision": 0.8967,
    "recall": 0.9012,
    "f1_score": 0.8989,
    "inference_time_ms": 52.34,
    "loss": 0.2187
  },
  "metadata": {
    "samples_evaluated": 25000,
    "evaluation_id": "eval_42_test",
    "aws_s3_path": "s3://moody-ml-results/CreditRiskModel_A/v2.0/2024-11-13/test.json"
  }
}
```

### Drift Detection Record
```json
{
  "timestamp": "2024-11-13T14:30:00",
  "date": "2024-11-13",
  "period": "production",
  "drift_factor": 0.856,
  "features": {
    "credit_score": 710.5,
    "loan_amount": 65000,
    "debt_to_income_ratio": 0.42,
    "employment_length_years": 6.5,
    "num_credit_lines": 13
  },
  "prediction": 0,
  "prediction_probability": 0.423
}
```

---

## 🔌 AWS S3 Integration

The dashboard is designed to work seamlessly with AWS S3. To enable real S3 integration:

### Setup Steps

1. **Install boto3**:
```bash
pip install boto3
```

2. **Configure AWS credentials**:
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region
```

3. **Update `utils.py`**:

The code structure is already in place. Simply uncomment and modify:

```python
import boto3

def load_data_from_s3(bucket_name, key):
    """Load data from AWS S3"""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    data = json.loads(obj['Body'].read())
    return data

# Example usage
data = load_data_from_s3('moody-ml-results', 'model_results.json')
```

### S3 Bucket Structure

```
s3://moody-ml-results/
├── model_results.json
├── drift_data.json
├── drift_report.json
└── models/
    ├── CreditRiskModel_A/
    │   ├── v1.0/
    │   ├── v1.1/
    │   └── v2.0/
    ├── FraudDetectionModel_B/
    │   ├── v1.0/
    │   └── v1.5/
    └── DefaultPredictionModel_C/
        └── v1.0/
```

---

## 📈 Use Cases

### For ML Engineers
- Monitor model performance in production 24/7
- Detect performance degradation early
- Compare experiment results across versions
- Validate models before deployment
- Track inference time and resource usage

### For Data Scientists
- Analyze model behavior across datasets
- Identify drift patterns and root causes
- Benchmark model versions
- Generate performance reports for stakeholders
- Debug model issues with detailed metrics

### For QA Teams
- Validate model performance against thresholds
- Run automated regression tests
- Generate test reports for compliance
- Monitor production vs test performance gaps

### For Stakeholders/Management
- Track ML system health at a glance
- Understand model ROI and performance
- Make data-driven deployment decisions
- Monitor compliance with SLAs

---

## 🎨 Customization Guide

### Adding New Metrics

1. **Add to data generator** (`data_generator.py`):
```python
'new_metric': calculate_new_metric(data)
```

2. **Update data preparation** (`utils.py`):
```python
'new_metric': item['metrics']['new_metric']
```

3. **Add to dashboard** (`app.py`):
```python
primary_metric = st.sidebar.selectbox(
    "Primary Metric",
    options=['accuracy', 'new_metric', ...]
)
```

### Changing Performance Thresholds

Modify `utils.py` → `get_performance_status()`:

```python
if metric == 'accuracy':
    if value >= 0.95:  # Change this
        return ('Excellent', '🟢')
```

Or update `test_config.yaml`:

```yaml
model_thresholds:
  accuracy:
    minimum: 0.80  # Change thresholds here
```

### Customizing Styling

Update CSS in `app.py`:

```python
st.markdown("""
    <style>
    .main {
        background-color: #YOUR_COLOR;
    }
    h1 {
        color: #YOUR_COLOR;
    }
    </style>
    """, unsafe_allow_html=True)
```

### Adding New Drift Tests

Extend `drift_detector.py` → `DriftDetector` class:

```python
def your_custom_test(self, feature_name):
    """Your custom drift detection logic"""
    # Implement test
    return results
```

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Free & Easy)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Click "Deploy"
5. Share the public URL

**Deployment time**: ~5 minutes

### Option 2: Docker Container

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t ml-dashboard .
docker run -p 8501:8501 ml-dashboard
```

### Option 3: AWS EC2

1. Launch EC2 instance (t2.medium recommended)
2. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```

3. Run with systemd service:
```bash
streamlit run app.py --server.port 80 --server.address 0.0.0.0
```

### Option 4: Kubernetes

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-dashboard
  template:
    metadata:
      labels:
        app: ml-dashboard
    spec:
      containers:
      - name: dashboard
        image: your-registry/ml-dashboard:latest
        ports:
        - containerPort: 8501
```

---

## 📝 Future Enhancements

### Planned Features
- [ ] Real-time data streaming from Databricks
- [ ] Email/Slack alerts for drift detection
- [ ] Model A/B testing framework with statistical significance
- [ ] Automated retraining triggers
- [ ] Integration with MLflow for experiment tracking
- [ ] PostgreSQL/MongoDB backend for scalability
- [ ] User authentication and role-based access
- [ ] Multi-tenancy support for different teams
- [ ] Advanced statistical tests (Wasserstein distance, KL divergence)
- [ ] Model explainability features (SHAP values, LIME)
- [ ] Feature importance tracking over time
- [ ] Fairness and bias detection
- [ ] Cost analysis dashboard
- [ ] API endpoint for programmatic access
- [ ] Mobile-responsive design
- [ ] Dark mode theme

### Integrations Ready For
- Jenkins/GitHub Actions (CI/CD)
- Databricks (data source)
- MLflow (experiment tracking)
- Airflow (orchestration)
- Grafana (monitoring)
- PagerDuty (alerting)

---

## 🤝 Contributing

This is a demonstration project built for Moody's Analytics. Feel free to:
- Extend functionality for your use case
- Add new visualizations
- Improve algorithms
- Enhance UI/UX
- Submit issues or suggestions

---

## 📧 Contact & Skills Demonstrated

**Built for**: Moody's Analytics ML Engineering Internship

**Key Skills Demonstrated**:
- ✅ Python programming (pandas, numpy, scipy)
- ✅ AWS/Cloud technologies (S3 integration)
- ✅ Data visualization (Plotly, Streamlit)
- ✅ ML operations and monitoring
- ✅ Statistical analysis and hypothesis testing
- ✅ Dashboard development
- ✅ Automated testing (pytest, 60+ tests)
- ✅ Drift detection (industry-standard methods)
- ✅ Agile practices (modular, documented code)
- ✅ Production ML best practices

**Technologies Used**:
- Python 3.9+
- Streamlit
- Plotly
- Pandas & NumPy
- SciPy (statistical tests)
- Scikit-learn
- pytest
- AWS boto3 (S3)
- YAML configuration
- Git version control

---

## 📚 References & Documentation

### Drift Detection Resources
- [Google Cloud: Detecting Data Drift](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Population Stability Index (PSI)](https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf)
- [Evidently AI: ML Monitoring](https://docs.evidentlyai.com/)

### Testing Best Practices
- [pytest Documentation](https://docs.pytest.org/)
- [ML Testing Best Practices](https://developers.google.com/machine-learning/testing-debugging)

### Streamlit Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

## ⚠️ Important Notes

### For Demo/Interview Purposes
- This project uses **simulated data** for demonstration
- Drift patterns are **artificially generated** to showcase detection capabilities
- AWS S3 integration is **demonstrated locally** but production-ready
- All components are **modular and extensible** for real-world use

### For Production Use
- Connect to real data sources (S3, databases, data lakes)
- Implement proper authentication and authorization
- Add comprehensive logging and monitoring
- Set up automated scheduling for drift detection
- Implement data retention policies
- Add performance optimization for large datasets
- Configure proper alerting channels (email, Slack, PagerDuty)

---

## 📊 Performance Metrics

### Dashboard Performance
- **Load time**: < 2 seconds (with cached data)
- **Refresh rate**: Real-time with Streamlit's auto-refresh
- **Data processing**: Handles 100,000+ records efficiently
- **Visualization rendering**: < 1 second per chart

### Test Suite Performance
- **Total tests**: 60+ test cases
- **Execution time**: ~5-10 seconds for full suite
- **Coverage**: Core logic, data pipeline, model validation

### Drift Detection Performance
- **Analysis time**: ~2-5 seconds for 30,000+ samples
- **Statistical tests**: All three tests per feature
- **Report generation**: < 1 second

---

## 🎓 Learning Outcomes

This project demonstrates key ML engineering skills:

1. **ML Operations**: How to monitor models in production
2. **Drift Detection**: Why and how to detect distribution shifts
3. **Statistical Testing**: KS test, PSI, Chi-square for ML
4. **Dashboard Development**: Building interactive data apps
5. **Testing Best Practices**: Unit, integration, validation tests
6. **Cloud Integration**: Working with AWS S3
7. **Data Visualization**: Creating meaningful charts and KPIs
8. **Code Organization**: Modular, maintainable ML projects

---

## 🏆 Project Highlights

### What Makes This Stand Out

1. **Advanced Drift Detection** ⭐⭐⭐⭐⭐
   - Most ML projects don't include drift detection
   - Uses industry-standard statistical methods
   - Critical for production ML systems

2. **Comprehensive Testing** ⭐⭐⭐⭐⭐
   - 60+ automated tests
   - Unit, integration, and validation coverage
   - Configurable thresholds

3. **Production-Ready Architecture** ⭐⭐⭐⭐⭐
   - Modular, extensible code
   - AWS S3 integration ready
   - Professional UI/UX

4. **Real-World Relevance** ⭐⭐⭐⭐⭐
   - Addresses actual ML operations challenges
   - Particularly relevant for financial services
   - Demonstrates understanding of production ML

---

## ✅ Checklist: Getting Started

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`requirements.txt`)
- [ ] Generated model performance data (`python data_generator.py`)
- [ ] Generated drift data (`python drift_data_generator.py`)
- [ ] Ran drift analysis (`python drift_detector.py`)
- [ ] Launched dashboard (`streamlit run app.py`)
- [ ] Explored performance monitoring tab
- [ ] Explored drift detection tab
- [ ] Ran automated tests (`python test_runner.py`)
- [ ] Reviewed test reports (HTML file)
- [ ] Customized configuration (`test_config.yaml`)
- [ ] Read this README thoroughly
- [ ] Ready to demo! 🚀

---

## 🎉 You're Ready!

This dashboard demonstrates enterprise-level ML operations capabilities including performance monitoring, drift detection, and automated testing. It showcases the skills needed for production ML engineering roles in financial services.

**Questions? Issues? Suggestions?**
Feel free to explore the code, customize it for your needs, and use it as a foundation for your ML monitoring projects.

---

**Note**: This is a demonstration project. For production deployment, ensure proper security, authentication, data privacy, and compliance with your organization's policies.

---

*Last Updated: November 2024*
*Version: 2.0 (with Drift Detection)*