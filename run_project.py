"""
Spotify Analytics - Complete Project Execution Script
Runs the entire end-to-end analytics pipeline
"""

import os
import sys
import subprocess
import time
from datetime import datetime


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_step(step_num, title):
    """Print formatted step"""
    print(f"\n[STEP {step_num}] {title}")
    print("-" * 50)


def run_script(script_path, description):
    """Run a Python script and handle errors"""
    try:
        print(f"[EXECUTING] {description}")
        start_time = time.time()
        
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        elapsed_time = time.time() - start_time
        print(f"[SUCCESS] Completed in {elapsed_time:.1f} seconds")
        
        if result.stdout:
            print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error executing {script_path}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return False


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 
        'sklearn', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # Special check for sqlite3 which is built-in
    try:
        import sqlite3
    except ImportError:
        missing_packages.append('sqlite3')
    
    if missing_packages:
        print(f"[ERROR] Missing required packages: {missing_packages}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("[SUCCESS] All required packages are installed")
    return True


def verify_data_file():
    """Verify that the input data file exists"""
    data_file = "data/spotify_history.csv"
    
    if not os.path.exists(data_file):
        print(f"[ERROR] Data file not found: {data_file}")
        print("Please ensure the spotify_history.csv file is in the data/ directory")
        return False
    
    # Check file size
    file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
    print(f"[SUCCESS] Data file found: {data_file} ({file_size:.1f} MB)")
    return True


def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'data', 'notebooks', 'sql', 'scripts', 'reports', 
        'dashboards', 'models', 'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("[SUCCESS] Directory structure verified")


def main():
    """Main execution function"""
    
    print_header("SPOTIFY ANALYTICS - END-TO-END PROJECT EXECUTION")
    print("Comprehensive Product Analytics Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Pre-execution checks
    print_step(0, "Pre-Execution Validation")
    
    if not check_requirements():
        return False
    
    if not verify_data_file():
        return False
    
    create_directory_structure()
    
    # Define execution pipeline
    pipeline_steps = [
        {
            'step': 1,
            'title': 'Data Preprocessing & Cleaning',
            'script': 'scripts/data_preprocessing.py',
            'description': 'Clean raw data, handle missing values, standardize formats'
        },
        {
            'step': 2,
            'title': 'Feature Engineering',
            'script': 'scripts/feature_engineering.py',
            'description': 'Create temporal, behavioral, and contextual features'
        },
        {
            'step': 3,
            'title': 'Machine Learning Pipeline',
            'script': 'scripts/model_pipeline.py',
            'description': 'Train skip prediction and user segmentation models'
        }
    ]
    
    # Execute pipeline
    start_time = time.time()
    successful_steps = 0
    
    for step in pipeline_steps:
        print_step(step['step'], step['title'])
        
        if run_script(step['script'], step['description']):
            successful_steps += 1
        else:
            print(f"[ERROR] Pipeline failed at step {step['step']}")
            break
    
    # Summary
    total_time = time.time() - start_time
    
    print_header("EXECUTION SUMMARY")
    print(f"Completed Steps: {successful_steps}/{len(pipeline_steps)}")
    print(f"Total Execution Time: {total_time/60:.1f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_steps == len(pipeline_steps):
        print("\n[SUCCESS] PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
        print("\nGenerated Outputs:")
        print("   - data/spotify_cleaned.csv - Cleaned dataset")
        print("   - data/spotify_features.csv - Feature-engineered dataset")
        print("   - models/ - Trained ML models")
        print("   - notebooks/ - Jupyter notebooks for exploration")
        print("   - sql/ - SQL queries for database analysis")
        
        print("\nNext Steps:")
        print("   1. Open notebooks/01_eda.ipynb for exploratory analysis")
        print("   2. Run SQL queries against your data warehouse")
        print("   3. Deploy models for real-time predictions")
        
        return True
    else:
        print(f"\n[ERROR] PROJECT EXECUTION FAILED at step {successful_steps + 1}")
        print("\nTroubleshooting:")
        print("   - Check error messages above")
        print("   - Verify data file format and content")
        print("   - Ensure all dependencies are installed")
        print("   - Check file permissions and disk space")
        
        return False


def run_specific_component(component):
    """Run a specific component of the pipeline"""
    
    components = {
        'preprocessing': {
            'script': 'scripts/data_preprocessing.py',
            'description': 'Data preprocessing and cleaning'
        },
        'features': {
            'script': 'scripts/feature_engineering.py',
            'description': 'Feature engineering'
        },
        'models': {
            'script': 'scripts/model_pipeline.py',
            'description': 'Machine learning pipeline'
        }
    }
    
    if component not in components:
        print(f"[ERROR] Unknown component: {component}")
        print(f"Available components: {list(components.keys())}")
        return False
    
    print_header(f"RUNNING COMPONENT: {component.upper()}")
    
    comp = components[component]
    return run_script(comp['script'], comp['description'])


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        component = sys.argv[1].lower()
        if component in ['preprocessing', 'features', 'models']:
            success = run_specific_component(component)
        elif component == 'help':
            print("Spotify Analytics Pipeline")
            print("\nUsage:")
            print("  python run_project.py           # Run complete pipeline")
            print("  python run_project.py preprocessing  # Run only preprocessing")
            print("  python run_project.py features      # Run only feature engineering")
            print("  python run_project.py models        # Run only ML pipeline")
            print("  python run_project.py help          # Show this help message")
            sys.exit(0)
        else:
            print(f"[ERROR] Unknown argument: {component}")
            print("Use 'python run_project.py help' for usage information")
            sys.exit(1)
    else:
        # Run complete pipeline
        success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
