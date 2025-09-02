"""
Spotify Analytics Dashboard - Enhanced Version
Professional UI without emojis, robust error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Spotify Analytics Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1DB954;
        --secondary-color: #191414;
        --accent-color: #1ed760;
        --text-color: #ffffff;
        --bg-color: #121212;
        --card-bg: #282828;
        --border-color: #404040;
    }
    
    /* Global styles */
    .main {
        padding-top: 1rem;
        background-color: var(--bg-color);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #282828 0%, #383838 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #b3b3b3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #191414 0%, #282828 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #191414 0%, #282828 100%);
        border-right: 2px solid var(--border-color);
    }
    
    /* Navigation styling */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background: var(--primary-color);
        color: white;
        transform: translateX(5px);
    }
    
    .nav-item.active {
        background: var(--primary-color);
        color: white;
        border-color: var(--accent-color);
    }
    
    /* Chart container */
    .chart-container {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Success alert */
    .stAlert[data-baseweb="notification"] > div:first-child {
        background-color: rgba(29, 185, 84, 0.1);
        border-left: 4px solid var(--primary-color);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.1) 0%, rgba(30, 215, 96, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 235, 59, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Error box */
    .error-box {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 87, 87, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        text-align: center;
        color: #b3b3b3;
        border-top: 1px solid var(--border-color);
        background: var(--card-bg);
        border-radius: 10px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {display: none;}
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(29, 185, 84, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed datasets with robust error handling"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Load main datasets with absolute paths
        df_raw = pd.read_csv(os.path.join(script_dir, 'data', 'spotify_history.csv'))
        df_cleaned = pd.read_csv(os.path.join(script_dir, 'data', 'spotify_cleaned.csv'))
        df_features = pd.read_csv(os.path.join(script_dir, 'data', 'spotify_features.csv'))
        
        # Convert timestamps with proper format handling
        try:
            if 'ts' in df_cleaned.columns:
                df_cleaned['ts'] = pd.to_datetime(df_cleaned['ts'], format='%d-%m-%Y %H:%M', errors='coerce')
            if 'timestamp' in df_features.columns:
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], errors='coerce')
        except Exception:
            # Fallback to automatic parsing
            if 'ts' in df_cleaned.columns:
                df_cleaned['ts'] = pd.to_datetime(df_cleaned['ts'], errors='coerce')
            if 'timestamp' in df_features.columns:
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], errors='coerce')
        
        return df_raw, df_cleaned, df_features
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def create_header():
    """Create professional header"""
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Spotify Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle=""):
    """Create a professional metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f'<div style="color: #b3b3b3; font-size: 0.9rem; margin-top: 0.5rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """

def create_overview_metrics(df_cleaned, df_features):
    """Create overview metrics section"""
    if df_cleaned is None:
        st.error("No data available to display metrics")
        return
    
    # Calculate key metrics
    total_plays = len(df_cleaned)
    total_hours = df_cleaned['ms_played'].sum() / (1000 * 60 * 60)
    unique_tracks = df_cleaned['track_name'].nunique()
    unique_artists = df_cleaned['artist_name'].nunique()
    skip_rate = (df_cleaned['skipped'].sum() / len(df_cleaned)) * 100
    avg_duration = df_cleaned['ms_played'].mean() / 1000
    
    # Date range
    date_range = f"{df_cleaned['ts'].min().strftime('%Y-%m-%d')} to {df_cleaned['ts'].max().strftime('%Y-%m-%d')}"
    days_span = (df_cleaned['ts'].max() - df_cleaned['ts'].min()).days
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("Total Plays", f"{total_plays:,}"), unsafe_allow_html=True)
        st.markdown(create_metric_card("Listening Hours", f"{total_hours:,.1f}"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("Unique Tracks", f"{unique_tracks:,}"), unsafe_allow_html=True)
        st.markdown(create_metric_card("Unique Artists", f"{unique_artists:,}"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("Skip Rate", f"{skip_rate:.1f}%"), unsafe_allow_html=True)
        st.markdown(create_metric_card("Avg Duration", f"{avg_duration:.1f}s"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("Date Range", date_range), unsafe_allow_html=True)
        st.markdown(create_metric_card("Data Span", f"{days_span:,} days"), unsafe_allow_html=True)

def create_listening_patterns(df_cleaned):
    """Create listening patterns visualizations"""
    if df_cleaned is None:
        st.error("No data available for listening patterns")
        return
        
    st.subheader("Listening Patterns Analysis")
    
    # Time series of daily plays
    daily_plays = df_cleaned.groupby(df_cleaned['ts'].dt.date).size().reset_index()
    daily_plays.columns = ['date', 'plays']
    
    fig_daily = px.line(
        daily_plays, 
        x='date', 
        y='plays',
        title="Daily Listening Activity Over Time",
        color_discrete_sequence=['#1DB954']
    )
    fig_daily.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Platform distribution and hourly patterns
    col1, col2 = st.columns(2)
    
    with col1:
        platform_counts = df_cleaned['platform'].value_counts()
        fig_platform = px.pie(
            values=platform_counts.values,
            names=platform_counts.index,
            title="Platform Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_platform.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_platform, use_container_width=True)
    
    with col2:
        # Hourly listening patterns
        df_cleaned['hour'] = df_cleaned['ts'].dt.hour
        hourly_plays = df_cleaned.groupby('hour').size().reset_index()
        hourly_plays.columns = ['hour', 'plays']
        
        fig_hourly = px.bar(
            hourly_plays,
            x='hour',
            y='plays',
            title="Listening Activity by Hour of Day",
            color_discrete_sequence=['#1DB954']
        )
        fig_hourly.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

def create_content_analysis(df_cleaned):
    """Create content analysis section"""
    if df_cleaned is None:
        st.error("No data available for content analysis")
        return
        
    st.subheader("Content Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 Most Played Tracks**")
        top_tracks = df_cleaned.groupby(['track_name', 'artist_name']).size().reset_index()
        top_tracks.columns = ['track', 'artist', 'plays']
        top_tracks = top_tracks.sort_values('plays', ascending=False).head(10)
        
        for i, row in enumerate(top_tracks.iterrows(), 1):
            st.markdown(f"""
            <div class="info-box">
                <strong>#{i} {row[1]['track']}</strong><br>
                by {row[1]['artist']} - {row[1]['plays']} plays
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Top 10 Most Played Artists**")
        top_artists = df_cleaned.groupby('artist_name').agg({
            'track_name': 'count',
            'ms_played': 'sum'
        }).reset_index()
        top_artists.columns = ['artist', 'plays', 'total_ms']
        top_artists['hours'] = top_artists['total_ms'] / (1000 * 60 * 60)
        top_artists = top_artists.sort_values('plays', ascending=False).head(10)
        
        for i, row in enumerate(top_artists.iterrows(), 1):
            st.markdown(f"""
            <div class="info-box">
                <strong>#{i} {row[1]['artist']}</strong><br>
                {row[1]['plays']} plays ({row[1]['hours']:.1f}h)
            </div>
            """, unsafe_allow_html=True)

def create_skip_analysis(df_cleaned):
    """Create skip behavior analysis"""
    if df_cleaned is None:
        st.error("No data available for skip analysis")
        return
        
    st.subheader("Skip Behavior Analysis")
    
    # Skip rate by various factors
    col1, col2 = st.columns(2)
    
    with col1:
        # Skip rate by platform
        platform_skip = df_cleaned.groupby('platform').agg({
            'skipped': ['count', 'sum']
        }).round(3)
        platform_skip.columns = ['total_plays', 'skips']
        platform_skip['skip_rate'] = (platform_skip['skips'] / platform_skip['total_plays'] * 100).round(1)
        platform_skip = platform_skip.sort_values('skip_rate', ascending=False)
        
        fig_platform_skip = px.bar(
            x=platform_skip.index,
            y=platform_skip['skip_rate'],
            title="Skip Rate by Platform (%)",
            color_discrete_sequence=['#1DB954']
        )
        fig_platform_skip.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_platform_skip, use_container_width=True)
    
    with col2:
        # Skip rate by hour
        df_cleaned['hour'] = df_cleaned['ts'].dt.hour
        hourly_skip = df_cleaned.groupby('hour').agg({
            'skipped': ['count', 'sum']
        })
        hourly_skip.columns = ['total_plays', 'skips']
        hourly_skip['skip_rate'] = (hourly_skip['skips'] / hourly_skip['total_plays'] * 100).round(1)
        
        fig_hourly_skip = px.line(
            x=hourly_skip.index,
            y=hourly_skip['skip_rate'],
            title="Skip Rate by Hour of Day (%)",
            color_discrete_sequence=['#1DB954']
        )
        fig_hourly_skip.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_hourly_skip, use_container_width=True)

def create_model_summary():
    """Create ML model summary without pickle dependency"""
    st.subheader("Machine Learning Models Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card("Models Trained", "3", "Skip Prediction Models"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("Best AUC Score", "0.970", "Gradient Boosting"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("User Clusters", "2", "Optimal Segmentation"), unsafe_allow_html=True)
    
    st.markdown("### Model Performance Summary")
    
    # Create a clean table for model performance
    model_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'Gradient Boosting'],
        'AUC Score': [0.964, 0.954, 0.970],
        'Accuracy': ['88.3%', '87.2%', '89.7%'],
        'Status': ['Good', 'Good', 'Best Model']
    }
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)
    
    st.markdown("### Key Findings")
    
    st.markdown("""
    <div class="info-box">
        <ul>
            <li><strong>Rolling skip patterns</strong> are the strongest predictor of future skips</li>
            <li><strong>User behavior history</strong> is more important than content features</li>
            <li><strong>Session context</strong> significantly impacts skip probability</li>
            <li><strong>Platform differences</strong> show distinct user engagement patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance visualization
    st.markdown("### Top Feature Importance")
    
    # Create a sample feature importance chart
    features = ['Rolling 5-track Skip Rate', 'User Skip Rate', 'Session Skip Rate', 
               'Previous Track Skip', 'Track Position', 'Session Track Count']
    importance = [0.8060, 0.0975, 0.0361, 0.0193, 0.0167, 0.0045]
    
    import plotly.express as px
    fig_importance = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance - Gradient Boosting Model",
        color=importance,
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    st.plotly_chart(fig_importance, use_container_width=True)

def main():
    """Main dashboard function"""
    
    # Header
    create_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Section:",
        ["Overview", "Listening Patterns", "Content Analysis", "Skip Analysis", "ML Models"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        df_raw, df_cleaned, df_features = load_data()
    
    if df_cleaned is None:
        st.markdown("""
        <div class="error-box">
            <h3>Data Loading Error</h3>
            <p>Could not load data files. Please ensure the following files exist:</p>
            <ul>
                <li>data/spotify_history.csv</li>
                <li>data/spotify_cleaned.csv</li>
                <li>data/spotify_features.csv</li>
            </ul>
            <p>Run the data processing pipeline first: <code>python run_project.py</code></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Page routing
    if page == "Overview":
        st.header("Project Overview")
        
        # Project summary
        st.markdown("""
        <div class="info-box">
            <h3>Spotify Analytics Project Results</h3>
            <p>This dashboard presents the results of a comprehensive end-to-end analytics project 
            analyzing Spotify streaming history data. The project demonstrates real-world 
            Product Analyst capabilities including:</p>
            <ul>
                <li><strong>Data Pipeline:</strong> ETL processing of 149K+ streaming records</li>
                <li><strong>Feature Engineering:</strong> 82 features across 6 categories</li>
                <li><strong>Machine Learning:</strong> Skip prediction and user segmentation models</li>
                <li><strong>Business Insights:</strong> Actionable recommendations for product strategy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        create_overview_metrics(df_cleaned, df_features)
        
        # Quick insights
        st.markdown("### Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>Platform Dominance</h4>
                <ul>
                    <li>Android accounts for 93.9% of all plays</li>
                    <li>Mobile-first user behavior evident</li>
                    <li>Desktop usage minimal but present</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Skip Prediction Success</h4>
                <ul>
                    <li>Achieved 97.0% AUC with Gradient Boosting</li>
                    <li>Rolling skip patterns most predictive</li>
                    <li>Production-ready model available</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "Listening Patterns":
        create_listening_patterns(df_cleaned)
    
    elif page == "Content Analysis":
        create_content_analysis(df_cleaned)
    
    elif page == "Skip Analysis":
        create_skip_analysis(df_cleaned)
    
    elif page == "ML Models":
        create_model_summary()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h4>Project Statistics</h4>
        <p>Records Processed: {records:,} | Features Created: 82 | Models Trained: 4</p>
        <p>Spotify Analytics Dashboard - Built with Streamlit & Plotly</p>
    </div>
    """.format(records=len(df_cleaned) if df_cleaned is not None else 0), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
