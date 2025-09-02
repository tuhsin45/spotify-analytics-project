# Spotify Streaming Analytics - End-to-End Product Analytics Project

## Project Overview
This project replicates a real-world case study for Product Analyst/Business Analyst roles using Spotify streaming history data. The goal is to extract actionable insights about user engagement, listening patterns, and behavioral trends to drive product decisions.

## Dataset Description
The dataset contains ~150k event-level streaming sessions with the following features:
- `spotify_track_uri`: Unique track identifier
- `ts`: Timestamp of the streaming event
- `platform`: Streaming platform (Android, iOS, web player)
- `ms_played`: Duration played in milliseconds
- `track_name`: Name of the track
- `artist_name`: Artist name
- `album_name`: Album name
- `reason_start`: Why the track started playing
- `reason_end`: Why the track stopped playing
- `shuffle`: Whether shuffle mode was on
- `skipped`: Whether the track was skipped

## Project Structure
```
spotify_analytics/
├── data/                    # Raw and processed datasets
│   └── spotify_history.csv
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_business_insights.ipynb  # Business Intelligence
│   └── 03_advanced_analytics.ipynb # ML models and clustering
├── sql/                    # SQL queries and analysis
│   ├── basic_queries.sql
│   ├── advanced_queries.sql
│   └── sessionization.sql
├── scripts/                # Python scripts for data processing
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_pipeline.py
└── README.md
```

## Objectives

### 1. Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Timestamp conversion and feature creation
- Listening pattern analysis by time, platform, device
- Skip behavior analysis
- Session length calculations
- Top tracks/artists/albums identification

### 2. Business/Product Insights
- Peak listening times identification
- Platform preference analysis
- Skip behavior drivers analysis
- Cross-platform engagement comparison
- Session start/end reason impact on retention

### 3. Advanced Analytics
- User sessionization implementation
- Retention and churn analysis
- Skip prediction modeling
- User/session clustering and segmentation

### 4. SQL Analysis
- Top tracks/artists by playtime queries
- Skip rate analysis by platform and time
- Average session length calculations
- Advanced SQL techniques (CTEs, window functions)

## Key Questions to Answer
1. When do users listen to music most frequently?
2. Which platforms have the highest engagement?
3. What factors drive users to skip songs?
4. How can we improve user retention?
5. What are the different user behavioral segments?

## Tools and Technologies
- **Python**: Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn
- **SQL**: PostgreSQL/SQLite for data querying
- **Visualization**: Interactive dashboards and static reports
- **Machine Learning**: Classification models for skip prediction, clustering for segmentation

## Getting Started
1. Install required dependencies: `pip install -r requirements.txt`
2. Run the EDA notebook: `01_eda.ipynb`
3. Execute SQL queries in the `sql/` directory
4. Review business insights in `02_business_insights.ipynb`
5. Explore advanced analytics in `03_advanced_analytics.ipynb`

## Expected Outcomes
- Comprehensive understanding of user listening behaviors
- Actionable recommendations for product improvements
- Predictive models for user engagement
- Executive-ready dashboard and reports
