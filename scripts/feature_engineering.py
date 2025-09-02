"""
Feature Engineering Script for Spotify Analytics
Creates advanced features for machine learning and analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SpotifyFeatureEngineer:
    """
    Advanced feature engineering for Spotify streaming data
    """
    
    def __init__(self, dataframe):
        """
        Initialize with preprocessed dataframe
        
        Args:
            dataframe (pd.DataFrame): Cleaned Spotify data
        """
        self.df = dataframe.copy()
        self.feature_log = []
    
    def log(self, message):
        """Add message to feature engineering log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.feature_log.append(log_entry)
        print(log_entry)
    
    def create_temporal_features(self):
        """Create comprehensive temporal features"""
        
        self.log("Creating temporal features...")
        
        # Basic temporal features
        self.df['hour_of_day'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        self.df['day_of_week_num'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['week_of_year'] = self.df['timestamp'].dt.isocalendar().week
        
        # Advanced temporal features
        self.df['is_weekend'] = self.df['day_of_week_num'].isin([5, 6])
        self.df['is_workday'] = self.df['day_of_week_num'].isin([0, 1, 2, 3, 4])
        
        # Time of day categories
        def categorize_time(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        self.df['time_of_day'] = self.df['hour_of_day'].apply(categorize_time)
        
        # Peak hours
        self.df['is_peak_hour'] = self.df['hour_of_day'].isin([17, 18, 19, 20])
        self.df['is_morning_commute'] = self.df['hour_of_day'].isin([7, 8, 9])
        self.df['is_evening_commute'] = self.df['hour_of_day'].isin([17, 18, 19])
        self.df['is_late_night'] = self.df['hour_of_day'].isin([23, 0, 1, 2, 3, 4, 5])
        
        # Seasonal features
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        self.df['season'] = self.df['month'].apply(get_season)
        
        self.log(f"Created {13} temporal features")
    
    def create_listening_behavior_features(self):
        """Create features related to listening behavior"""
        
        self.log("Creating listening behavior features...")
        
        # Convert durations
        self.df['seconds_played'] = self.df['ms_played'] / 1000
        self.df['minutes_played'] = self.df['seconds_played'] / 60
        
        # Estimate track lengths and completion rates
        track_lengths = self.df.groupby('spotify_track_uri')['ms_played'].max().reset_index()
        track_lengths.rename(columns={'ms_played': 'estimated_track_length_ms'}, inplace=True)
        self.df = self.df.merge(track_lengths, on='spotify_track_uri', how='left')
        
        # Calculate percent played
        self.df['percent_played'] = np.where(
            self.df['estimated_track_length_ms'] > 0,
            (self.df['ms_played'] / self.df['estimated_track_length_ms']) * 100,
            0
        )
        # Cap at 100%
        self.df['percent_played'] = np.minimum(self.df['percent_played'], 100)
        
        # Track length categories
        self.df['track_length_seconds'] = self.df['estimated_track_length_ms'] / 1000
        self.df['track_length_category'] = pd.cut(
            self.df['track_length_seconds'],
            bins=[0, 120, 180, 240, 300, float('inf')],
            labels=['<2min', '2-3min', '3-4min', '4-5min', '5min+']
        )
        
        # Skip indicators
        self.df['is_skip'] = (
            (self.df['skipped'] == True) | 
            (self.df['percent_played'] < 30) |
            (self.df['reason_end'].isin(['nextbtn', 'backbtn']))
        )
        
        # Listening quality
        self.df['listening_quality'] = np.where(
            self.df['percent_played'] >= 80, 'High',
            np.where(self.df['percent_played'] >= 50, 'Medium', 'Low')
        )
        
        # Engagement scores
        self.df['completion_score'] = self.df['percent_played'] / 100
        self.df['engagement_score'] = (
            self.df['completion_score'] * 0.6 + 
            (1 - self.df['is_skip'].astype(int)) * 0.4
        )
        
        self.log(f"Created {11} listening behavior features")
    
    def create_user_behavior_features(self):
        """Create user-level behavioral features"""
        
        self.log("Creating user behavior features...")
        
        # Using spotify_track_uri as user proxy for this analysis
        user_stats = self.df.groupby('spotify_track_uri').agg({
            'timestamp': ['count', 'min', 'max'],
            'is_skip': 'mean',
            'percent_played': 'mean',
            'engagement_score': 'mean',
            'shuffle': 'mean',
            'platform': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'seconds_played': 'sum'
        }).round(3)
        
        user_stats.columns = ['user_play_count', 'user_first_play', 'user_last_play',
                             'user_skip_rate', 'user_avg_completion', 'user_avg_engagement',
                             'user_shuffle_rate', 'user_preferred_platform', 'user_total_seconds']
        
        # User lifecycle features
        user_stats['user_lifespan_days'] = (
            user_stats['user_last_play'] - user_stats['user_first_play']
        ).dt.days + 1
        
        user_stats['user_frequency'] = user_stats['user_play_count'] / user_stats['user_lifespan_days']
        
        # User segments
        user_stats['user_segment'] = np.where(
            (user_stats['user_play_count'] >= 20) & (user_stats['user_avg_engagement'] >= 0.7), 'Power User',
            np.where(
                (user_stats['user_play_count'] >= 10) & (user_stats['user_avg_engagement'] >= 0.5), 'Engaged User',
                np.where(
                    user_stats['user_skip_rate'] >= 0.6, 'At-Risk User',
                    'Casual User'
                )
            )
        )
        
        # Merge back to main dataframe
        user_features = user_stats[['user_play_count', 'user_skip_rate', 'user_avg_completion',
                                   'user_avg_engagement', 'user_segment', 'user_frequency']].reset_index()
        
        self.df = self.df.merge(user_features, on='spotify_track_uri', how='left')
        
        self.log(f"Created {6} user behavior features")
    
    def create_content_features(self):
        """Create content-related features"""
        
        self.log("Creating content features...")
        
        # Track popularity features
        track_stats = self.df.groupby(['track_name', 'artist_name']).agg({
            'spotify_track_uri': 'count',
            'is_skip': 'mean',
            'percent_played': 'mean',
            'engagement_score': 'mean'
        }).round(3)
        
        track_stats.columns = ['track_play_count', 'track_skip_rate', 
                              'track_avg_completion', 'track_avg_engagement']
        
        track_stats.reset_index(inplace=True)
        self.df = self.df.merge(track_stats, on=['track_name', 'artist_name'], how='left')
        
        # Artist popularity features
        artist_stats = self.df.groupby('artist_name').agg({
            'spotify_track_uri': 'count',
            'track_name': 'nunique',
            'is_skip': 'mean',
            'engagement_score': 'mean'
        }).round(3)
        
        artist_stats.columns = ['artist_play_count', 'artist_unique_tracks',
                               'artist_skip_rate', 'artist_avg_engagement']
        
        artist_stats.reset_index(inplace=True)
        self.df = self.df.merge(artist_stats, on='artist_name', how='left')
        
        # Content categories
        self.df['track_popularity'] = pd.cut(
            self.df['track_play_count'],
            bins=[0, 5, 20, 50, float('inf')],
            labels=['Rare', 'Uncommon', 'Popular', 'Very Popular']
        )
        
        self.df['artist_popularity'] = pd.cut(
            self.df['artist_play_count'],
            bins=[0, 10, 50, 200, float('inf')],
            labels=['Niche', 'Emerging', 'Mainstream', 'Superstar']
        )
        
        self.log(f"Created {8} content features")
    
    def create_session_features(self):
        """Create session-based features"""
        
        self.log("Creating session features...")
        
        # Sort by timestamp for session analysis
        self.df = self.df.sort_values('timestamp')
        
        # Calculate time gaps between consecutive plays
        self.df['time_since_previous'] = self.df['timestamp'].diff()
        
        # Session breaks (30-minute rule)
        session_break_threshold = timedelta(minutes=30)
        self.df['is_session_start'] = (
            (self.df['time_since_previous'] > session_break_threshold) | 
            (self.df['time_since_previous'].isna())
        )
        
        # Assign session IDs
        self.df['session_id'] = self.df['is_session_start'].cumsum()
        
        # Session-level statistics
        session_stats = self.df.groupby('session_id').agg({
            'timestamp': ['min', 'max', 'count'],
            'is_skip': 'mean',
            'engagement_score': 'mean',
            'platform': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'seconds_played': 'sum'
        }).round(3)
        
        session_stats.columns = ['session_start', 'session_end', 'session_track_count',
                                'session_skip_rate', 'session_avg_engagement',
                                'session_platform', 'session_total_seconds']
        
        # Session duration
        session_stats['session_duration_minutes'] = (
            (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 60
        ).round(1)
        
        # Session quality
        session_stats['session_quality'] = np.where(
            (session_stats['session_track_count'] >= 10) & 
            (session_stats['session_skip_rate'] <= 0.3) &
            (session_stats['session_avg_engagement'] >= 0.7), 'High',
            np.where(
                (session_stats['session_track_count'] >= 5) & 
                (session_stats['session_skip_rate'] <= 0.5), 'Medium',
                'Low'
            )
        )
        
        # Merge session features back
        session_features = session_stats[['session_track_count', 'session_skip_rate',
                                         'session_avg_engagement', 'session_duration_minutes',
                                         'session_quality']].reset_index()
        
        self.df = self.df.merge(session_features, on='session_id', how='left')
        
        # Position in session
        self.df['track_position_in_session'] = self.df.groupby('session_id').cumcount() + 1
        
        self.log(f"Created {6} session features")
    
    def create_context_features(self):
        """Create contextual features"""
        
        self.log("Creating context features...")
        
        # Platform switching behavior
        self.df['prev_platform'] = self.df['platform'].shift(1)
        self.df['platform_switch'] = (self.df['platform'] != self.df['prev_platform']).fillna(False)
        
        # Reason patterns
        self.df['is_autoplay'] = self.df['reason_start'] == 'autoplay'
        self.df['is_manual_start'] = self.df['reason_start'].isin(['clickrow', 'playbtn'])
        self.df['is_manual_skip'] = self.df['reason_end'].isin(['nextbtn', 'backbtn'])
        self.df['is_natural_end'] = self.df['reason_end'] == 'trackdone'
        
        # Listening patterns
        self.df['is_binge_session'] = self.df['session_track_count'] >= 20
        self.df['is_short_session'] = self.df['session_track_count'] <= 3
        
        # Platform preferences by time
        platform_time_preference = self.df.groupby(['time_of_day', 'platform']).size().unstack(fill_value=0)
        platform_time_pref_norm = platform_time_preference.div(platform_time_preference.sum(axis=1), axis=0)
        
        # Diversity metrics
        self.df['artist_diversity_in_session'] = self.df.groupby('session_id')['artist_name'].transform('nunique')
        
        self.log(f"Created {9} context features")
    
    def create_advanced_ml_features(self):
        """Create advanced features for machine learning"""
        
        self.log("Creating advanced ML features...")
        
        # Interaction features
        self.df['hour_platform_interaction'] = self.df['hour_of_day'].astype(str) + '_' + self.df['platform']
        self.df['weekend_platform_interaction'] = self.df['is_weekend'].astype(str) + '_' + self.df['platform']
        
        # Lag features (previous track information)
        self.df['prev_track_skip'] = self.df['is_skip'].shift(1).fillna(False)
        self.df['prev_track_completion'] = self.df['percent_played'].shift(1).fillna(0)
        
        # Rolling statistics (last 5 tracks)
        self.df['rolling_5_skip_rate'] = self.df['is_skip'].rolling(5, min_periods=1).mean()
        self.df['rolling_5_avg_completion'] = self.df['percent_played'].rolling(5, min_periods=1).mean()
        
        # Categorical encoding for ML
        self.df['platform_encoded'] = pd.Categorical(self.df['platform']).codes
        self.df['time_of_day_encoded'] = pd.Categorical(self.df['time_of_day']).codes
        self.df['listening_quality_encoded'] = pd.Categorical(self.df['listening_quality']).codes
        
        # Polynomial features for key metrics
        self.df['completion_squared'] = self.df['percent_played'] ** 2
        self.df['track_length_log'] = np.log1p(self.df['track_length_seconds'])
        
        self.log(f"Created {10} advanced ML features")
    
    def create_summary_statistics(self):
        """Create summary statistics for the feature set"""
        
        # Count different types of features
        temporal_features = [col for col in self.df.columns if any(x in col.lower() for x in ['hour', 'day', 'time', 'season', 'weekend', 'morning', 'evening', 'night', 'peak'])]
        behavioral_features = [col for col in self.df.columns if any(x in col.lower() for x in ['skip', 'completion', 'engagement', 'listening', 'quality'])]
        user_features = [col for col in self.df.columns if col.startswith('user_')]
        content_features = [col for col in self.df.columns if any(x in col.lower() for x in ['track_', 'artist_', 'album_'])]
        session_features = [col for col in self.df.columns if col.startswith('session_')]
        
        feature_summary = {
            'total_features': len(self.df.columns),
            'temporal_features': len(temporal_features),
            'behavioral_features': len(behavioral_features),
            'user_features': len(user_features),
            'content_features': len(content_features),
            'session_features': len(session_features),
            'total_records': len(self.df)
        }
        
        self.log("\n" + "="*50)
        self.log("FEATURE ENGINEERING SUMMARY")
        self.log("="*50)
        
        for key, value in feature_summary.items():
            self.log(f"{key.replace('_', ' ').title()}: {value}")
        
        # Data types summary
        dtype_summary = self.df.dtypes.value_counts()
        self.log(f"\nData Types Distribution:")
        for dtype, count in dtype_summary.items():
            self.log(f"  {dtype}: {count} columns")
        
        return feature_summary
    
    def run_feature_engineering(self, output_file_path=None):
        """
        Run the complete feature engineering pipeline
        
        Args:
            output_file_path (str): Path to save feature-engineered data (optional)
            
        Returns:
            pandas.DataFrame: Feature-engineered dataframe
        """
        
        self.log("Starting feature engineering pipeline...")
        
        # Create different categories of features
        self.create_temporal_features()
        self.create_listening_behavior_features()
        self.create_user_behavior_features()
        self.create_content_features()
        self.create_session_features()
        self.create_context_features()
        self.create_advanced_ml_features()
        
        # Generate summary
        summary = self.create_summary_statistics()
        
        # Save engineered data
        if output_file_path:
            try:
                self.df.to_csv(output_file_path, index=False)
                self.log(f"Feature-engineered data saved to: {output_file_path}")
            except Exception as e:
                self.log(f"Error saving feature-engineered data: {str(e)}")
        
        self.log("Feature engineering pipeline completed successfully!")
        
        return self.df
    
    def get_feature_log(self):
        """Return the feature engineering log"""
        return self.feature_log


def main():
    """Main function to run feature engineering"""
    
    # File paths
    input_file = 'data/spotify_cleaned.csv'
    output_file = 'data/spotify_features.csv'
    
    # Load cleaned data
    try:
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df):,} records from {input_file}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Initialize feature engineer
    feature_engineer = SpotifyFeatureEngineer(df)
    
    # Run feature engineering
    featured_df = feature_engineer.run_feature_engineering(output_file)
    
    if featured_df is not None:
        print(f"\n[SUCCESS] Feature engineering completed successfully!")
        print(f"[INFO] Final dataset: {len(featured_df):,} records with {len(featured_df.columns)} features")
        print(f"[INFO] Saved to: {output_file}")
        
        # Save feature engineering log
        log_file = 'data/feature_engineering_log.txt'
        with open(log_file, 'w', encoding='utf-8') as f:
            for entry in feature_engineer.get_feature_log():
                f.write(entry + '\n')
        print(f"[INFO] Log saved to: {log_file}")
        
        # Display feature categories
        print(f"\n[INFO] Created features across multiple categories:")
        print(f"   - Temporal features (time-based patterns)")
        print(f"   - Behavioral features (listening habits)")
        print(f"   - User features (user-level statistics)")
        print(f"   - Content features (track/artist metrics)")
        print(f"   - Session features (session-level analysis)")
        print(f"   - Context features (situational factors)")
        print(f"   - ML features (advanced modeling features)")
    
    else:
        print("[ERROR] Feature engineering failed!")


if __name__ == "__main__":
    main()
