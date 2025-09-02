"""
Data Preprocessing Script for Spotify Analytics
Handles data cleaning, validation, and basic transformations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SpotifyDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for Spotify streaming data
    """
    
    def __init__(self, input_file_path):
        """
        Initialize preprocessor with input file path
        
        Args:
            input_file_path (str): Path to the raw CSV file
        """
        self.input_file_path = input_file_path
        self.df = None
        self.preprocessing_log = []
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(self.input_file_path)
            self.log(f"Successfully loaded {len(self.df):,} records from {self.input_file_path}")
            return True
        except Exception as e:
            self.log(f"Error loading data: {str(e)}")
            return False
    
    def log(self, message):
        """Add message to preprocessing log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.preprocessing_log.append(log_entry)
        print(log_entry)
    
    def validate_data_structure(self):
        """Validate that required columns exist"""
        required_columns = [
            'spotify_track_uri', 'ts', 'platform', 'ms_played',
            'track_name', 'artist_name', 'album_name',
            'reason_start', 'reason_end', 'shuffle', 'skipped'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            self.log(f"ERROR: Missing required columns: {missing_columns}")
            return False
        
        self.log("All required columns present")
        return True
    
    def clean_missing_values(self):
        """Handle missing values in the dataset"""
        initial_rows = len(self.df)
        
        # Check missing values
        missing_summary = self.df.isnull().sum()
        self.log(f"Missing values summary:\n{missing_summary[missing_summary > 0]}")
        
        # Remove rows with missing critical values
        critical_columns = ['spotify_track_uri', 'ts', 'ms_played']
        before_drop = len(self.df)
        self.df = self.df.dropna(subset=critical_columns)
        dropped_critical = before_drop - len(self.df)
        
        if dropped_critical > 0:
            self.log(f"Removed {dropped_critical} rows with missing critical values")
        
        # Fill missing categorical values
        categorical_columns = ['track_name', 'artist_name', 'album_name', 'reason_start', 'reason_end']
        for col in categorical_columns:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    self.df[col] = self.df[col].fillna('Unknown')
                    self.log(f"Filled {missing_count} missing values in {col} with 'Unknown'")
        
        self.log(f"Data cleaning completed. Rows: {initial_rows} -> {len(self.df)}")
    
    def standardize_data_types(self):
        """Standardize data types across columns"""
        
        # Convert boolean columns
        boolean_columns = ['shuffle', 'skipped']
        for col in boolean_columns:
            if col in self.df.columns:
                # Handle various boolean representations
                self.df[col] = self.df[col].astype(str).str.upper()
                self.df[col] = self.df[col].map({'TRUE': True, 'FALSE': False, 'T': True, 'F': False})
                
                # Fill any remaining NaN with False
                self.df[col] = self.df[col].fillna(False)
                self.log(f"Standardized {col} to boolean type")
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['ms_played']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.log(f"Converted {col} to numeric type")
        
        # Clean and standardize text columns
        text_columns = ['track_name', 'artist_name', 'album_name', 'platform', 'reason_start', 'reason_end']
        for col in text_columns:
            if col in self.df.columns:
                # Remove extra whitespace and standardize
                self.df[col] = self.df[col].astype(str).str.strip()
                # Replace empty strings with 'Unknown'
                self.df[col] = self.df[col].replace('', 'Unknown')
                self.log(f"Cleaned text formatting in {col}")
    
    def parse_timestamps(self):
        """Parse and validate timestamp data"""
        try:
            # Parse timestamp
            self.df['timestamp'] = pd.to_datetime(self.df['ts'], format='%d-%m-%Y %H:%M', errors='coerce')
            
            # Check for invalid timestamps
            invalid_timestamps = self.df['timestamp'].isnull().sum()
            if invalid_timestamps > 0:
                self.log(f"WARNING: {invalid_timestamps} invalid timestamps found, removing these rows")
                self.df = self.df.dropna(subset=['timestamp'])
            
            # Validate timestamp range
            min_date = self.df['timestamp'].min()
            max_date = self.df['timestamp'].max()
            date_range = (max_date - min_date).days
            
            self.log(f"Timestamp range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range} days)")
            
            # Flag suspicious timestamps (future dates or too old)
            future_threshold = datetime.now() + timedelta(days=1)
            past_threshold = datetime(2008, 1, 1)  # Spotify founded in 2008
            
            future_count = (self.df['timestamp'] > future_threshold).sum()
            old_count = (self.df['timestamp'] < past_threshold).sum()
            
            if future_count > 0:
                self.log(f"WARNING: {future_count} timestamps in the future")
            if old_count > 0:
                self.log(f"WARNING: {old_count} timestamps before 2008")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR parsing timestamps: {str(e)}")
            return False
    
    def detect_and_handle_outliers(self):
        """Detect and handle outliers in numeric columns"""
        
        # Analyze ms_played outliers
        q1 = self.df['ms_played'].quantile(0.25)
        q3 = self.df['ms_played'].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_count = ((self.df['ms_played'] < lower_bound) | 
                         (self.df['ms_played'] > upper_bound)).sum()
        
        self.log(f"Detected {outliers_count} outliers in ms_played using IQR method")
        
        # Cap extreme values (beyond 15 minutes = 900,000 ms)
        extreme_values = (self.df['ms_played'] > 900000).sum()
        if extreme_values > 0:
            self.df.loc[self.df['ms_played'] > 900000, 'ms_played'] = 900000
            self.log(f"Capped {extreme_values} extremely high ms_played values to 15 minutes")
        
        # Remove negative values
        negative_values = (self.df['ms_played'] < 0).sum()
        if negative_values > 0:
            self.df = self.df[self.df['ms_played'] >= 0]
            self.log(f"Removed {negative_values} rows with negative ms_played values")
    
    def remove_duplicates(self):
        """Remove duplicate records"""
        initial_count = len(self.df)
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates()
        exact_duplicates = initial_count - len(self.df)
        
        if exact_duplicates > 0:
            self.log(f"Removed {exact_duplicates} exact duplicate rows")
        
        # Check for suspicious duplicates (same track, same minute)
        self.df['timestamp_minute'] = self.df['timestamp'].dt.floor('min')
        
        suspicious_duplicates = self.df.duplicated(
            subset=['spotify_track_uri', 'timestamp_minute'], keep=False
        ).sum()
        
        if suspicious_duplicates > 0:
            self.log(f"Found {suspicious_duplicates} potential duplicate plays (same track, same minute)")
        
        # Clean up temporary column
        self.df = self.df.drop(columns=['timestamp_minute'])
    
    def validate_business_logic(self):
        """Validate business logic constraints"""
        
        # Check for valid platforms
        valid_platforms = ['web player', 'iOS', 'Android', 'desktop']
        invalid_platforms = self.df[~self.df['platform'].isin(valid_platforms)]
        
        if len(invalid_platforms) > 0:
            unique_invalid = invalid_platforms['platform'].unique()
            self.log(f"WARNING: Found {len(invalid_platforms)} records with unexpected platforms: {list(unique_invalid)}")
        
        # Check skip consistency
        skipped_with_full_play = self.df[(self.df['skipped'] == True) & (self.df['ms_played'] > 240000)]
        if len(skipped_with_full_play) > 0:
            self.log(f"WARNING: {len(skipped_with_full_play)} tracks marked as skipped but played >4 minutes")
        
        # Check reason_start and reason_end consistency
        reason_start_counts = self.df['reason_start'].value_counts()
        reason_end_counts = self.df['reason_end'].value_counts()
        
        self.log(f"Top 5 reason_start: {dict(reason_start_counts.head())}")
        self.log(f"Top 5 reason_end: {dict(reason_end_counts.head())}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary of the cleaned data"""
        
        summary = {
            'total_records': len(self.df),
            'date_range': f"{self.df['timestamp'].min().strftime('%Y-%m-%d')} to {self.df['timestamp'].max().strftime('%Y-%m-%d')}",
            'unique_tracks': self.df['spotify_track_uri'].nunique(),
            'unique_artists': self.df['artist_name'].nunique(),
            'total_listening_hours': round(self.df['ms_played'].sum() / 3600000, 1),
            'platform_distribution': dict(self.df['platform'].value_counts()),
            'skip_rate': f"{(self.df['skipped'].mean() * 100):.1f}%",
            'avg_play_duration_seconds': round(self.df['ms_played'].mean() / 1000, 1)
        }
        
        self.log("\n" + "="*50)
        self.log("DATA SUMMARY REPORT")
        self.log("="*50)
        
        for key, value in summary.items():
            self.log(f"{key.replace('_', ' ').title()}: {value}")
        
        return summary
    
    def run_full_preprocessing(self, output_file_path=None):
        """
        Run the complete preprocessing pipeline
        
        Args:
            output_file_path (str): Path to save cleaned data (optional)
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        
        self.log("Starting Spotify data preprocessing pipeline...")
        
        # Step 1: Load data
        if not self.load_data():
            return None
        
        # Step 2: Validate structure
        if not self.validate_data_structure():
            return None
        
        # Step 3: Clean missing values
        self.clean_missing_values()
        
        # Step 4: Standardize data types
        self.standardize_data_types()
        
        # Step 5: Parse timestamps
        if not self.parse_timestamps():
            return None
        
        # Step 6: Handle outliers
        self.detect_and_handle_outliers()
        
        # Step 7: Remove duplicates
        self.remove_duplicates()
        
        # Step 8: Validate business logic
        self.validate_business_logic()
        
        # Step 9: Generate summary
        summary = self.generate_summary_report()
        
        # Step 10: Save cleaned data
        if output_file_path:
            try:
                self.df.to_csv(output_file_path, index=False)
                self.log(f"Cleaned data saved to: {output_file_path}")
            except Exception as e:
                self.log(f"Error saving cleaned data: {str(e)}")
        
        self.log("Preprocessing pipeline completed successfully!")
        
        return self.df
    
    def get_preprocessing_log(self):
        """Return the preprocessing log"""
        return self.preprocessing_log


def main():
    """Main function to run preprocessing"""
    
    # File paths
    input_file = 'data/spotify_history.csv'
    output_file = 'data/spotify_cleaned.csv'
    
    # Initialize preprocessor
    preprocessor = SpotifyDataPreprocessor(input_file)
    
    # Run preprocessing
    cleaned_df = preprocessor.run_full_preprocessing(output_file)
    
    if cleaned_df is not None:
        print(f"\n[SUCCESS] Preprocessing completed successfully!")
        print(f"[INFO] Final dataset: {len(cleaned_df):,} records")
        print(f"[INFO] Saved to: {output_file}")
        
        # Save preprocessing log
        log_file = 'data/preprocessing_log.txt'
        with open(log_file, 'w', encoding='utf-8') as f:
            for entry in preprocessor.get_preprocessing_log():
                f.write(entry + '\n')
        print(f"[INFO] Log saved to: {log_file}")
    
    else:
        print("[ERROR] Preprocessing failed!")


if __name__ == "__main__":
    main()
