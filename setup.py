"""
Setup script for initializing the project
Run this once after installing dependencies
"""
import os
import sys
import shutil
import argparse

def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    
    directories = [
        'data/processed/train',
        'data/processed/test',
        'models',
        'results/explanations',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")

def verify_dependencies():
    """Verify that key dependencies are installed"""
    print("\nVerifying dependencies...")
    
    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'cv2',
        'pandas',
        'numpy',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (MISSING)")
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def verify_data_structure():
    """Verify MUSTARD++ data structure"""
    print("\nVerifying data structure...")

    metadata_candidates = ['data/metadata.xlsx', 'data/metadata.csv']
    metadata_file = None
    for candidate in metadata_candidates:
        if os.path.exists(candidate):
            metadata_file = candidate
            break

    required_files = [
        'data/context_videos',
        'data/utterance_videos'
    ]
    
    all_exist = True
    if metadata_file:
        print(f"✓ {metadata_file}")
    else:
        print("✗ data/metadata.xlsx or data/metadata.csv (MISSING)")
        all_exist = False

    for item in required_files:
        if os.path.exists(item):
            print(f"✓ {item}")
        else:
            print(f"✗ {item} (MISSING)")
            all_exist = False
    
    if all_exist:
        # Count files
        if metadata_file:
            import pandas as pd
            if metadata_file.endswith('.xlsx'):
                df = pd.read_excel(metadata_file)
            else:
                df = pd.read_csv(metadata_file)
            print(f"\n  CSV records: {len(df)}")
            print(f"  Columns: {', '.join(df.columns.tolist())}")
        
        context_videos = len([f for f in os.listdir('data/context_videos') if f.endswith('.mp4')])
        utterance_videos = len([f for f in os.listdir('data/utterance_videos') if f.endswith('.mp4')])
        print(f"  Context videos: {context_videos}")
        print(f"  Utterance videos: {utterance_videos}")
        
        return True
    else:
        print("\n⚠ Please ensure MUSTARD++ dataset is in data/ directory")
        return False

def create_sample_config():
    """Create sample configuration if needed"""
    print("\nChecking configuration...")
    
    if os.path.exists('config/config.yaml'):
        print("✓ config/config.yaml exists")
        return True
    else:
        print("✗ config/config.yaml not found")
        return False

def print_summary(deps_ok, data_ok, config_ok):
    """Print setup summary"""
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)
    
    print(f"Dependencies:      {'✓ OK' if deps_ok else '✗ MISSING'}")
    print(f"Data Structure:    {'✓ OK' if data_ok else '✗ MISSING'}")
    print(f"Configuration:     {'✓ OK' if config_ok else '✗ MISSING'}")
    
    if deps_ok and data_ok and config_ok:
        print("\n✓ Setup complete! You can now run:")
        print("  python train.py")
    else:
        print("\n⚠ Please fix the issues above before running training")

def main():
    parser = argparse.ArgumentParser(description='Setup Sarcasm Detection Framework')
    parser.add_argument('--skip-verification', action='store_true', help='Skip verification steps')
    args = parser.parse_args()
    
    print("=" * 70)
    print("SARCASM DETECTION FRAMEWORK - SETUP")
    print("=" * 70 + "\n")
    
    # Setup directories
    setup_directories()
    
    if not args.skip_verification:
        # Verify everything
        deps_ok = verify_dependencies()
        data_ok = verify_data_structure()
        config_ok = create_sample_config()
        
        # Print summary
        print_summary(deps_ok, data_ok, config_ok)
    else:
        print("\nSkipped verification steps")

if __name__ == "__main__":
    main()

