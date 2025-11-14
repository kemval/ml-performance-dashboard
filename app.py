"""
ML Model Performance Monitoring Dashboard

This module implements a Streamlit-based dashboard for monitoring machine learning model performance
and detecting data/concept drift. The dashboard provides interactive visualizations of model metrics,
performance trends, and drift detection results to help ML teams maintain model health and performance.

Key Features:
- Real-time performance metrics visualization (accuracy, precision, recall, F1, inference time)
- Model comparison and benchmarking
- Data drift detection and visualization
- Concept drift analysis
- Interactive charts and performance alerts
"""
import json
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data_from_json, 
    prepare_dataframe, 
    calculate_performance_change,
    get_best_model,
    detect_performance_degradation,
    get_performance_status,
    export_to_csv
)

# Page configuration
st.set_page_config(
    page_title="ML Model Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Moody's-style professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1e3a8a;
    }
    h2, h3 {
        color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 ML Model Performance Dashboard")
st.markdown("### Real-time monitoring and benchmarking of ML models")

# Data loading functions (cached, outside tabs)
@st.cache_data
def load_data():
    data = load_data_from_json()
    return prepare_dataframe(data)

@st.cache_data
def load_drift_report():
    """Load drift analysis report if available"""
    drift_path = 'data/drift_report.json'
    if os.path.exists(drift_path):
        with open(drift_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_drift_data():
    """Load drift data for visualization"""
    drift_path = 'data/drift_data.json'
    if os.path.exists(drift_path):
        with open(drift_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return None

# Create tabs for different views
tab1, tab2 = st.tabs(["📈 Performance Monitoring", "🔬 Drift Detection"])

# ============================================================================
# TAB 1: PERFORMANCE MONITORING
# ============================================================================
with tab1:
    st.markdown("### Performance Monitoring & Model Benchmarking")
    st.markdown("Track model accuracy, precision, recall, and inference metrics across multiple versions and datasets")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("❌ No data found! Please run `python data_generator.py` first to generate mock data.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    min_date = df['evaluation_date'].min().date()
    max_date = df['evaluation_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['evaluation_date'].dt.date >= start_date) & 
                (df['evaluation_date'].dt.date <= end_date)]
    
    # Model filter
    all_models = sorted(df['model_full_name'].unique())
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=all_models,
        default=all_models[:3] if len(all_models) >= 3 else all_models
    )
    
    if selected_models:
        df = df[df['model_full_name'].isin(selected_models)]
    
    # Dataset filter
    selected_datasets = st.sidebar.multiselect(
        "Select Datasets",
        options=['train', 'validation', 'test', 'production'],
        default=['test', 'production']
    )
    
    if selected_datasets:
        df = df[df['dataset'].isin(selected_datasets)]
    
    # Metric selector
    primary_metric = st.sidebar.selectbox(
        "Primary Metric",
        options=['accuracy', 'precision', 'recall', 'f1_score', 'loss', 'inference_time_ms'],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Data Source")
    st.sidebar.info("Simulates AWS S3 integration\n\n`s3://moody-ml-results/`")
    
    # Check if filtered data is empty
    if df.empty:
        st.warning("⚠️ No data matches your filters. Please adjust the filters.")
        st.stop()
    
    # Key Metrics Row
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_accuracy = df['accuracy'].mean()
        status, icon = get_performance_status(avg_accuracy, 'accuracy')
        st.metric(
            label=f"{icon} Average Accuracy",
            value=f"{avg_accuracy:.2%}",
            delta=f"{calculate_performance_change(df, selected_models[0] if selected_models else all_models[0], 'accuracy')}%"
        )
    
    with col2:
        avg_f1 = df['f1_score'].mean()
        st.metric(
            label="📊 Average F1 Score",
            value=f"{avg_f1:.4f}",
            delta=f"{calculate_performance_change(df, selected_models[0] if selected_models else all_models[0], 'f1_score')}%"
        )
    
    with col3:
        avg_inference = df['inference_time_ms'].mean()
        status, icon = get_performance_status(avg_inference, 'inference_time_ms')
        st.metric(
            label=f"{icon} Avg Inference Time",
            value=f"{avg_inference:.1f} ms",
            delta=f"{calculate_performance_change(df, selected_models[0] if selected_models else all_models[0], 'inference_time_ms')}%",
            delta_color="inverse"
        )
    
    with col4:
        best_model_info = get_best_model(df, primary_metric, 'test')
        if best_model_info:
            st.metric(
                label="🏆 Best Model (Test)",
                value=best_model_info['model'].split('_')[-1],
                delta=f"{best_model_info['score']:.2%}"
            )
    
    st.markdown("---")
    
    # Performance Over Time
    st.markdown("### 📉 Performance Trends Over Time")
    
    # Create time series plot
    fig_time = go.Figure()
    
    for model in selected_models:
        model_data = df[df['model_full_name'] == model]
        avg_by_date = model_data.groupby('evaluation_date')[primary_metric].mean().reset_index()
        
        fig_time.add_trace(go.Scatter(
            x=avg_by_date['evaluation_date'],
            y=avg_by_date[primary_metric],
            mode='lines+markers',
            name=model,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig_time.update_layout(
        title=f"{primary_metric.replace('_', ' ').title()} Over Time",
        xaxis_title="Date",
        yaxis_title=primary_metric.replace('_', ' ').title(),
        hovermode='x unified',
        height=400,
        showlegend=True,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Two column layout for comparisons
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 🔄 Model Comparison")
        
        # Average metrics by model
        comparison_df = df.groupby('model_full_name').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean'
        }).reset_index()
        
        fig_comparison = go.Figure()
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics_to_compare:
            fig_comparison.add_trace(go.Bar(
                name=metric.title(),
                x=comparison_df['model_full_name'],
                y=comparison_df[metric],
                text=comparison_df[metric].round(3),
                textposition='auto',
            ))
        
        fig_comparison.update_layout(
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Score",
            height=400,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col_right:
        st.markdown("### 🎯 Performance by Dataset")
        
        # Performance across datasets
        dataset_perf = df.groupby(['dataset', 'model_full_name'])[primary_metric].mean().reset_index()
        
        fig_dataset = px.bar(
            dataset_perf,
            x='dataset',
            y=primary_metric,
            color='model_full_name',
            barmode='group',
            title=f"Average {primary_metric.replace('_', ' ').title()} by Dataset",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_dataset, use_container_width=True)
    
    # Performance Heatmap
    st.markdown("### 🗺️ Model Performance Heatmap")
    
    heatmap_data = df.groupby(['model_full_name', 'dataset'])[primary_metric].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='model_full_name', columns='dataset', values=primary_metric)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='RdYlGn',
        text=heatmap_pivot.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title=primary_metric.title())
    ))
    
    fig_heatmap.update_layout(
        title=f"{primary_metric.replace('_', ' ').title()} Heatmap: Model vs Dataset",
        xaxis_title="Dataset",
        yaxis_title="Model",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Alerts and Insights
    st.markdown("### 🚨 Alerts & Insights")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.markdown("#### ⚠️ Performance Alerts")
        alerts_found = False
        
        for model in selected_models:
            if detect_performance_degradation(df, model, primary_metric):
                st.warning(f"📉 **{model}**: Performance degradation detected in {primary_metric}")
                alerts_found = True
        
        if not alerts_found:
            st.success("✅ All models performing within expected ranges")
    
    with alert_col2:
        st.markdown("#### 💡 Key Insights")
        
        # Best performing model
        best = get_best_model(df, primary_metric, 'test')
        if best:
            st.info(f"🏆 **Top Performer**: {best['model']} ({best['score']:.2%})")
        
        # Production vs Test gap
        prod_avg = df[df['dataset'] == 'production'][primary_metric].mean() if 'production' in df['dataset'].values else 0
        test_avg = df[df['dataset'] == 'test'][primary_metric].mean() if 'test' in df['dataset'].values else 0
        
        if prod_avg > 0 and test_avg > 0:
            gap = ((test_avg - prod_avg) / test_avg) * 100
            if gap > 5:
                st.warning(f"⚠️ **Production Gap**: {gap:.1f}% lower than test performance")
            else:
                st.success(f"✅ **Production Gap**: Only {gap:.1f}% difference from test")
    
    # Data Export
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📥 Download Filtered Data (CSV)"):
            csv_file = export_to_csv(df)
            st.success(f"✅ Exported to {csv_file}")
    
    with col_export2:
        st.markdown(f"**Total Records**: {len(df)}")
    
    with col_export3:
        st.markdown(f"**Date Range**: {len(df['evaluation_date'].dt.date.unique())} days")

# ============================================================================
# TAB 2: DRIFT DETECTION
# ============================================================================
with tab2:
    st.markdown("### Data & Concept Drift Monitoring")
    st.markdown("Detect distribution shifts in input features and model predictions")
    st.markdown("---")
    
    # Load drift data
    drift_report = load_drift_report()
    drift_data = load_drift_data()
    
    if drift_report is None:
        st.warning("""
        ⚠️ **Drift Analysis Not Available**
        
        To enable drift detection:
        1. Run `python drift_data_generator.py` to generate feature data
        2. Run `python drift_detector.py` to perform drift analysis
        3. Refresh this page
        """)
    else:
        # Drift Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        
        summary = drift_report['drift_analysis']['summary']
        alerts = drift_report['alerts']
        concept_drift = drift_report['drift_analysis']['concept_drift']
        
        with col1:
            st.metric(
                label="🔍 Features Monitored",
                value=summary['total_features_analyzed']
            )
        
        with col2:
            drift_count = summary['features_with_drift']
            drift_icon = "🔴" if drift_count > 2 else "🟡" if drift_count > 0 else "🟢"
            st.metric(
                label=f"{drift_icon} Features with Drift",
                value=drift_count,
                delta=f"{summary['drift_percentage']:.0f}% of total"
            )
        
        with col3:
            concept_status = "Detected" if concept_drift['drift_detected'] else "Stable"
            concept_icon = "🔴" if concept_drift['drift_detected'] else "🟢"
            st.metric(
                label=f"{concept_icon} Concept Drift",
                value=concept_status,
                delta=f"{concept_drift['change_percent']:.1f}% change"
            )
        
        with col4:
            alert_count = len(alerts)
            alert_icon = "🚨" if alert_count > 3 else "⚠️" if alert_count > 0 else "✅"
            st.metric(
                label=f"{alert_icon} Active Alerts",
                value=alert_count
            )
        
        st.markdown("---")
        
        # Alerts Section
        if len(alerts) > 0:
            st.markdown("### 🚨 Drift Alerts")
            
            for alert in alerts:
                severity_color = {
                    'low': 'warning',
                    'moderate': 'warning', 
                    'high': 'error'
                }.get(alert['severity'], 'info')
                
                severity_icon = {
                    'low': '🟡',
                    'moderate': '🟠',
                    'high': '🔴'
                }.get(alert['severity'], '⚪')
                
                if severity_color == 'error':
                    st.error(f"{severity_icon} **[{alert['severity'].upper()}]** {alert['message']}\n\n→ {alert['recommendation']}")
                else:
                    st.warning(f"{severity_icon} **[{alert['severity'].upper()}]** {alert['message']}\n\n→ {alert['recommendation']}")
        else:
            st.success("✅ No significant drift detected across all features")
        
        st.markdown("---")
        
        # Feature Drift Analysis
        st.markdown("### 📊 Feature-Level Drift Analysis")
        
        feature_drift = drift_report['drift_analysis']['feature_drift']
        
        # Create drift summary table
        drift_summary_data = []
        for feature, data in feature_drift.items():
            drift_summary_data.append({
                'Feature': feature,
                'Drift Detected': '🔴 Yes' if data['overall_drift_detected'] else '🟢 No',
                'Confidence': f"{data['drift_confidence']*100:.0f}%",
                'Mean Shift': f"{data['statistics']['mean_shift_percent']:+.1f}%",
                'KS Test p-value': data['tests']['ks_test']['p_value'],
                'PSI': data['tests']['psi']['psi_value']
            })
        
        drift_summary_df = pd.DataFrame(drift_summary_data)
        st.dataframe(drift_summary_df, use_container_width=True)
        
        st.markdown("---")
        
        # Distribution Comparisons
        st.markdown("### 📉 Distribution Comparisons")
        
        if drift_data is not None:
            # Select feature to visualize
            selected_feature = st.selectbox(
                "Select feature to visualize",
                options=list(feature_drift.keys())
            )
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### Reference vs Production Distribution")
                
                # Extract feature data
                ref_data = drift_data[drift_data['period'] == 'reference']
                prod_data = drift_data[drift_data['period'] == 'production']
                
                # Extract feature values
                ref_values = [d[selected_feature] for d in ref_data['features']]
                prod_values = [d[selected_feature] for d in prod_data['features']]
                
                # Create histogram
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=ref_values,
                    name='Reference',
                    opacity=0.7,
                    marker_color='blue',
                    nbinsx=30
                ))
                
                fig_hist.add_trace(go.Histogram(
                    x=prod_values,
                    name='Production',
                    opacity=0.7,
                    marker_color='red',
                    nbinsx=30
                ))
                
                fig_hist.update_layout(
                    barmode='overlay',
                    title=f'{selected_feature} Distribution',
                    xaxis_title=selected_feature,
                    yaxis_title='Frequency',
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_right:
                st.markdown("#### Statistical Tests")
                
                feature_tests = feature_drift[selected_feature]['tests']
                stats_info = feature_drift[selected_feature]['statistics']
                
                st.markdown(f"""
                **Distribution Statistics:**
                - Reference Mean: `{stats_info['reference_mean']:.2f}`
                - Production Mean: `{stats_info['production_mean']:.2f}`
                - Mean Shift: `{stats_info['mean_shift_percent']:+.1f}%`
                
                **KS Test:**
                - Statistic: `{feature_tests['ks_test']['statistic']}`
                - P-value: `{feature_tests['ks_test']['p_value']}`
                - Drift: `{'🔴 Yes' if feature_tests['ks_test']['drift_detected'] else '🟢 No'}`
                
                **Population Stability Index (PSI):**
                - PSI Value: `{feature_tests['psi']['psi_value']}`
                - Threshold: `{feature_tests['psi']['threshold']}`
                - Severity: `{feature_tests['psi']['severity']}`
                
                **Chi-Square Test:**
                - Statistic: `{feature_tests['chi_square']['statistic']}`
                - P-value: `{feature_tests['chi_square']['p_value']}`
                - Drift: `{'🔴 Yes' if feature_tests['chi_square']['drift_detected'] else '🟢 No'}`
                """)
        
        st.markdown("---")
        
        # Concept Drift Visualization
        st.markdown("### 🎯 Concept Drift Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Prediction Distribution Shift")
            
            concept_data = drift_report['drift_analysis']['concept_drift']
            
            fig_concept = go.Figure()
            
            categories = ['Reference', 'Production']
            approval_rates = [
                concept_data['reference_approval_rate'],
                concept_data['production_approval_rate']
            ]
            
            fig_concept.add_trace(go.Bar(
                x=categories,
                y=approval_rates,
                text=[f"{rate:.1%}" for rate in approval_rates],
                textposition='auto',
                marker_color=['blue', 'red']
            ))
            
            fig_concept.update_layout(
                title='Model Approval Rate Comparison',
                yaxis_title='Approval Rate',
                height=400,
                template="plotly_white",
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig_concept, use_container_width=True)
        
        with col2:
            st.markdown("#### Concept Drift Details")
            
            st.markdown(f"""
            **Approval Rate Change:**
            - Reference Period: `{concept_data['reference_approval_rate']:.1%}`
            - Production Period: `{concept_data['production_approval_rate']:.1%}`
            - Change: `{concept_data['change_percent']:+.1f}%`
            
            **Statistical Significance:**
            - P-value: `{concept_data['p_value']}`
            - Drift Detected: `{'🔴 Yes' if concept_data['drift_detected'] else '🟢 No'}`
            - Severity: `{concept_data['severity']}`
            
            **Interpretation:**
            """)
            
            if concept_data['drift_detected']:
                if abs(concept_data['change_percent']) > 10:
                    st.error("⚠️ **Significant concept drift detected!** Model behavior has substantially changed. Immediate investigation recommended.")
                else:
                    st.warning("⚠️ **Moderate concept drift detected.** Monitor closely and consider retraining.")
            else:
                st.success("✅ **No significant concept drift.** Model predictions remain stable.")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if len(alerts) == 0:
            st.success("""
            ✅ **System Healthy**
            - Continue monitoring drift metrics
            - Maintain current model in production
            - Schedule next drift analysis
            """)
        elif len(alerts) <= 2:
            st.info("""
            ℹ️ **Minor Drift Detected**
            - Monitor affected features closely
            - Investigate data collection changes
            - Consider adding feature monitoring alerts
            - Schedule drift re-analysis in 1 week
            """)
        else:
            st.error("""
            🚨 **Action Required**
            - **Immediate**: Investigate root causes of drift
            - **Short-term**: Implement data quality checks
            - **Consider**: Model retraining with recent data
            - **Review**: Data pipeline and preprocessing steps
            - **Alert**: Notify ML engineering team
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>Built for Moody's Analytics | ML Model Performance Monitoring & Drift Detection</p>
    <p style='font-size: 0.9em;'>Features: Performance Tracking | Data Drift Detection | Concept Drift Analysis | AWS S3 Ready</p>
</div>
""", unsafe_allow_html=True)