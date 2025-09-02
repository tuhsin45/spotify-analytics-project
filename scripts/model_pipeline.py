"""
Machine Learning Pipeline for Spotify Analytics
Implements skip prediction and user segmentation models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib


class SpotifyMLPipeline:
    """
    Complete machine learning pipeline for Spotify analytics
    """
    
    def __init__(self, dataframe):
        """
        Initialize with feature-engineered dataframe
        
        Args:
            dataframe (pd.DataFrame): Feature-engineered Spotify data
        """
        self.df = dataframe.copy()
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.ml_log = []
    
    def log(self, message):
        """Add message to ML log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.ml_log.append(log_entry)
        print(log_entry)
    
    def prepare_skip_prediction_features(self):
        """Prepare features for skip prediction model"""
        
        self.log("Preparing features for skip prediction...")
        
        # Select relevant features for skip prediction
        skip_features = [
            # Temporal features
            'hour_of_day', 'day_of_week_num', 'is_weekend', 'is_peak_hour', 'is_late_night',
            
            # Track characteristics
            'track_length_seconds', 'track_play_count', 'track_skip_rate',
            
            # User behavior
            'user_skip_rate', 'user_avg_completion', 'user_frequency',
            
            # Session context
            'session_track_count', 'track_position_in_session', 'session_skip_rate',
            
            # Platform and context
            'platform_encoded', 'time_of_day_encoded', 'shuffle',
            
            # Content features
            'artist_skip_rate', 'artist_play_count',
            
            # Behavioral patterns
            'is_autoplay', 'is_manual_start', 'prev_track_skip', 'rolling_5_skip_rate'
        ]
        
        # Filter for available features
        available_features = [col for col in skip_features if col in self.df.columns]
        
        # Handle missing values
        X = self.df[available_features].fillna(0)
        y = self.df['is_skip'].astype(int)
        
        self.log(f"Selected {len(available_features)} features for skip prediction")
        self.log(f"Target distribution: {y.mean():.1%} skip rate")
        
        return X, y, available_features
    
    def train_skip_prediction_models(self):
        """Train multiple models for skip prediction"""
        
        self.log("Training skip prediction models...")
        
        # Prepare data
        X, y, feature_names = self.prepare_skip_prediction_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for LogisticRegression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['skip_prediction'] = scaler
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            )
        }
        
        # Train and evaluate models
        for name, model in models.items():
            self.log(f"Training {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            if name == 'Logistic Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Store results
            self.models[f'skip_prediction_{name}'] = model
            self.results[f'skip_prediction_{name}'] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'cv_auc': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'y_test': y_test,
                'feature_names': feature_names
            }
            
            self.log(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}, CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Select best model
        best_model_name = max(self.results.keys(), 
                             key=lambda k: self.results[k]['auc'] if 'skip_prediction' in k else 0)
        
        self.log(f"Best skip prediction model: {best_model_name.split('_')[-2:]} (AUC: {self.results[best_model_name]['auc']:.3f})")
        
        return self.results
    
    def analyze_feature_importance(self, model_name):
        """Analyze feature importance for tree-based models"""
        
        if model_name not in self.models:
            self.log(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            feature_names = self.results[model_name]['feature_names']
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.log(f"Top 10 features for {model_name}:")
            for _, row in importance_df.head(10).iterrows():
                self.log(f"  {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
        else:
            self.log(f"Feature importance not available for {model_name}")
            return None
    
    def prepare_clustering_features(self):
        """Prepare features for user clustering"""
        
        self.log("Preparing features for user clustering...")
        
        # Create user-level aggregations
        user_features = self.df.groupby('spotify_track_uri').agg({
            'timestamp': 'count',  # Total plays
            'seconds_played': ['sum', 'mean'],  # Total and average listening time
            'is_skip': 'mean',  # Skip rate
            'percent_played': 'mean',  # Average completion
            'engagement_score': 'mean',  # Average engagement
            'shuffle': 'mean',  # Shuffle usage
            'platform': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Preferred platform
            'time_of_day': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],  # Preferred time
            'session_id': 'nunique',  # Number of sessions
            'artist_name': 'nunique',  # Artist diversity
            'track_name': 'nunique'  # Track diversity
        }).round(3)
        
        user_features.columns = [
            'total_plays', 'total_seconds', 'avg_seconds_per_play',
            'skip_rate', 'avg_completion', 'avg_engagement', 'shuffle_rate',
            'preferred_platform', 'preferred_time', 'total_sessions',
            'artist_diversity', 'track_diversity'
        ]
        
        # Add derived features
        user_features['total_minutes'] = user_features['total_seconds'] / 60
        user_features['avg_session_length'] = user_features['total_plays'] / user_features['total_sessions']
        user_features['listening_intensity'] = user_features['total_minutes'] / user_features['total_sessions']
        
        # Filter for users with sufficient data
        user_features = user_features[user_features['total_plays'] >= 5]
        
        self.log(f"Created clustering features for {len(user_features):,} users")
        
        return user_features
    
    def perform_user_clustering(self, n_clusters_range=(2, 8)):
        """Perform K-means clustering for user segmentation"""
        
        self.log("Performing user clustering...")
        
        # Prepare clustering data
        user_features = self.prepare_clustering_features()
        
        # Select numerical features for clustering
        clustering_features = [
            'total_plays', 'avg_seconds_per_play', 'skip_rate',
            'avg_completion', 'avg_engagement', 'shuffle_rate',
            'avg_session_length', 'artist_diversity', 'listening_intensity'
        ]
        
        X_cluster = user_features[clustering_features].fillna(0)
        
        # Scale features
        cluster_scaler = StandardScaler()
        X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)
        self.scalers['clustering'] = cluster_scaler
        
        # Find optimal number of clusters
        k_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
        silhouette_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_cluster_scaled)
            silhouette_scores.append(silhouette_score(X_cluster_scaled, cluster_labels))
            inertias.append(kmeans.inertia_)
        
        # Select optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        self.log(f"Optimal number of clusters: {optimal_k} (Silhouette Score: {best_silhouette:.3f})")
        
        # Final clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        user_features['cluster'] = final_kmeans.fit_predict(X_cluster_scaled)
        
        # Store results
        self.models['user_clustering'] = final_kmeans
        self.results['user_clustering'] = {
            'user_features': user_features,
            'cluster_features': clustering_features,
            'optimal_k': optimal_k,
            'silhouette_score': best_silhouette,
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'k_range': k_range
        }
        
        # Analyze clusters
        cluster_analysis = user_features.groupby('cluster').agg({
            'total_plays': ['count', 'mean'],
            'skip_rate': 'mean',
            'avg_completion': 'mean',
            'avg_engagement': 'mean',
            'total_minutes': 'mean',
            'artist_diversity': 'mean'
        }).round(3)
        
        self.log("Cluster Analysis:")
        self.log(str(cluster_analysis))
        
        return user_features, cluster_analysis
    
    def evaluate_models(self):
        """Generate comprehensive model evaluation"""
        
        self.log("Generating model evaluation report...")
        
        evaluation_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'skip_prediction_models': {},
            'clustering_results': {}
        }
        
        # Skip prediction evaluation
        for model_name, results in self.results.items():
            if 'skip_prediction' in model_name:
                model_type = model_name.split('_')[-2:]
                evaluation_report['skip_prediction_models'][' '.join(model_type)] = {
                    'accuracy': results['accuracy'],
                    'auc_score': results['auc'],
                    'cv_auc_mean': results['cv_auc'],
                    'cv_auc_std': results['cv_std']
                }
        
        # Clustering evaluation
        if 'user_clustering' in self.results:
            clustering_results = self.results['user_clustering']
            evaluation_report['clustering_results'] = {
                'optimal_clusters': clustering_results['optimal_k'],
                'silhouette_score': clustering_results['silhouette_score'],
                'total_users_clustered': len(clustering_results['user_features'])
            }
        
        return evaluation_report
    
    def save_models(self, save_directory='../models/'):
        """Save trained models and scalers"""
        
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.log(f"Saving models to {save_directory}...")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_directory, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            self.log(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_directory, f"scaler_{scaler_name}.pkl")
            joblib.dump(scaler, scaler_path)
            self.log(f"Saved {scaler_name} scaler to {scaler_path}")
        
        # Save results
        results_path = os.path.join(save_directory, "model_results.pkl")
        joblib.dump(self.results, results_path)
        self.log(f"Saved model results to {results_path}")
    
    def create_prediction_pipeline(self, model_name, new_data):
        """Create prediction pipeline for new data"""
        
        if model_name not in self.models:
            self.log(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        feature_names = self.results[model_name]['feature_names']
        
        # Prepare features
        X_new = new_data[feature_names].fillna(0)
        
        # Scale if needed
        if 'Logistic Regression' in model_name and 'skip_prediction' in self.scalers:
            X_new = self.scalers['skip_prediction'].transform(X_new)
        
        # Make predictions
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)[:, 1]
        
        return predictions, probabilities
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        
        self.log("Starting complete ML pipeline...")
        
        # Train skip prediction models
        skip_results = self.train_skip_prediction_models()
        
        # Analyze feature importance for best model
        best_skip_model = max([k for k in self.results.keys() if 'skip_prediction' in k], 
                             key=lambda k: self.results[k]['auc'])
        
        if 'Random Forest' in best_skip_model or 'Gradient' in best_skip_model:
            feature_importance = self.analyze_feature_importance(best_skip_model)
        
        # Perform user clustering
        user_features, cluster_analysis = self.perform_user_clustering()
        
        # Generate evaluation report
        evaluation = self.evaluate_models()
        
        # Save models
        self.save_models()
        
        self.log("Complete ML pipeline finished successfully!")
        
        return {
            'skip_prediction_results': skip_results,
            'user_clustering': {'user_features': user_features, 'cluster_analysis': cluster_analysis},
            'evaluation': evaluation
        }
    
    def get_ml_log(self):
        """Return the ML log"""
        return self.ml_log


def main():
    """Main function to run ML pipeline"""
    
    # File paths
    input_file = 'data/spotify_features.csv'
    
    # Load feature-engineered data
    try:
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df):,} records from {input_file}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Initialize ML pipeline
    ml_pipeline = SpotifyMLPipeline(df)
    
    # Run complete pipeline
    results = ml_pipeline.run_complete_pipeline()
    
    if results:
        print(f"\n[SUCCESS] ML pipeline completed successfully!")
        
        # Print summary
        print(f"\n[INFO] RESULTS SUMMARY:")
        print(f"   - Skip Prediction Models: {len(results['skip_prediction_results'])} trained")
        print(f"   - User Clusters: {results['evaluation']['clustering_results']['optimal_clusters']} identified")
        print(f"   - Clustering Silhouette Score: {results['evaluation']['clustering_results']['silhouette_score']:.3f}")
        
        # Save ML log  
        log_file = 'data/ml_pipeline_log.txt'
        with open(log_file, 'w', encoding='utf-8') as f:
            for entry in ml_pipeline.get_ml_log():
                f.write(entry + '\n')
        print(f"[INFO] ML log saved to: {log_file}")
        
        # Save evaluation report
        import json
        eval_file = 'reports/model_evaluation.json'
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(results['evaluation'], f, indent=2)
        print(f"[INFO] Evaluation report saved to: {eval_file}")
    
    else:
        print("[ERROR] ML pipeline failed!")


if __name__ == "__main__":
    main()
