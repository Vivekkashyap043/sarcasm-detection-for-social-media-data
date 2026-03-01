#!/usr/bin/env python
"""
Quick reference script - Run this to see available commands
"""

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   MULTIMODAL SARCASM DETECTION FRAMEWORK v1.0.0              ║
    ║   Production-Grade Framework for Sarcasm Detection            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

def print_commands():
    commands = {
        "🚀 SETUP & INSTALLATION": [
            ("python setup.py", "Initialize project directories and verify setup"),
            ("pip install -r requirements.txt", "Install all dependencies"),
            ("pip install torch... --index-url ...", "Install CPU PyTorch (see INSTALLATION.md)"),
        ],
        "📚 DOCUMENTATION": [
            ("README.md", "Complete project documentation"),
            ("QUICKSTART.md", "5-minute quick start guide"),
            ("INSTALLATION.md", "Detailed installation guide"),
            ("ARCHITECTURE.md", "Technical architecture reference"),
            ("SOCIAL_MEDIA_INTEGRATION.md", "Social media deployment guide"),
            ("PROJECT_SUMMARY.md", "Project overview and summary"),
        ],
        "🎓 TRAINING & TESTING": [
            ("python train.py", "Train model with default settings"),
            ("python train.py --config config/config.yaml", "Train with custom config"),
            ("python test.py test --model models/best_model.pth", "Evaluate on test set"),
            ("python test.py explain --model models/best_model.pth", "Generate explanations"),
            ("python test.py predict --text '...' --video path.mp4 --model models/best_model.pth", "Predict on custom sample"),
        ],
        "📊 EXPECTED RESULTS": [
            ("results/evaluation_results.json", "Complete metrics (JSON)"),
            ("results/evaluation_results.csv", "Predictions and probabilities"),
            ("results/evaluation_report.txt", "Formatted evaluation report"),
            ("results/explanations/", "Explanation files (LIME/SHAP)"),
            ("logs/training.log", "Detailed training logs"),
        ],
        "⚙️ CONFIGURATION": [
            ("config/config.yaml", "Main configuration file"),
            ("Fine-tune batch_size", "For your hardware resources"),
            ("Select model architecture", "LSTM, Transformer, MLP, or Attention"),
            ("Adjust learning_rate", "For convergence speed"),
            ("Set num_epochs", "Number of training epochs"),
        ],
        "🔧 TROUBLESHOOTING": [
            ("setup.py --skip-verification", "Quick setup without checks"),
            ("Check logs/training.log", "For detailed error messages"),
            ("Reduce batch_size to 2-4", "If out of memory"),
            ("Use multimodal_mlp", "If training is too slow"),
            ("See INSTALLATION.md", "For installation issues"),
        ],
    }
    
    for category, items in commands.items():
        print(f"\n{category}")
        print("=" * 60)
        for item in items:
            if len(item) == 2:
                cmd, desc = item
                print(f"  {cmd:<45} # {desc}")
            else:
                print(f"  {item[0]}")

def print_file_structure():
    structure = """
    📁 PROJECT STRUCTURE
    ════════════════════════════════════════════════════════
    ├── 📄 Documentation
    │   ├── README.md                      (Full documentation)
    │   ├── QUICKSTART.md                  (Quick start)
    │   ├── INSTALLATION.md                (Setup guide)
    │   ├── ARCHITECTURE.md                (Technical details)
    │   ├── SOCIAL_MEDIA_INTEGRATION.md    (Social media guide)
    │   └── PROJECT_SUMMARY.md             (Overview)
    │
    ├── 📚 Source Code (src/)
    │   ├── utils.py                       (Utility functions)
    │   ├── data_preprocessing.py          (Data handling)
    │   ├── feature_extraction.py          (Feature extraction)
    │   ├── model.py                       (Model architectures)
    │   ├── training.py                    (Training logic)
    │   ├── evaluation.py                  (Evaluation metrics)
    │   └── explainability.py              (LIME/SHAP explanations)
    │
    ├── 🚀 Main Scripts
    │   ├── train.py                       (Training script)
    │   ├── test.py                        (Testing/prediction)
    │   └── setup.py                       (Setup helper)
    │
    ├── 📊 Data Directory
    │   ├── metadata.csv                   (Annotations)
    │   ├── context_videos/                (Context videos)
    │   ├── utterance_videos/              (Utterance videos)
    │   └── processed/                     (Auto-generated splits)
    │
    ├── 🤖 Models (auto-generated)
    │   └── best_model_*.pth               (Saved models)
    │
    ├── 📈 Results (auto-generated)
    │   ├── evaluation_results.json        (Metrics JSON)
    │   ├── evaluation_results.csv         (Predictions)
    │   ├── evaluation_report.txt          (Report)
    │   └── explanations/                  (Explanations)
    │
    └── 📝 Configuration & Dependencies
        ├── config/
        │   └── config.yaml                (Model config)
        ├── requirements.txt               (Dependencies)
        └── requirements-dev.txt           (Dev dependencies)
    """
    print(structure)

def print_quick_start():
    quick_start = """
    ⚡ 5-MINUTE QUICK START
    ════════════════════════════════════════════════════════
    
    1️⃣  SETUP
        python -m venv venv
        venv\\Scripts\\activate  # Windows
        pip install -r requirements.txt
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    2️⃣  VERIFY
        python setup.py
    
    3️⃣  TRAIN
        python train.py
    
    4️⃣  EVALUATE
        python test.py test --model models/best_model.pth
    
    5️⃣  RESULTS
        cat results/evaluation_report.txt
    
    📖 For detailed guide: See QUICKSTART.md
    """
    print(quick_start)

def print_features():
    features = """
    ✨ KEY FEATURES
    ════════════════════════════════════════════════════════
    
    ✓ Data Processing
      • Automatic 70/30 train/test split (stratified)
      • MUSTARD++ dataset support
      • Video frame extraction
      • Text preprocessing
    
    ✓ Feature Extraction
      • ResNet50 video features (2048-dim)
      • DistilBERT text embeddings (768-dim)
      • CPU-optimized models
    
    ✓ Model Architectures
      • LSTM (Recommended)
      • Transformer with Attention
      • Simple MLP (Fast)
      • Explicit Attention Fusion
    
    ✓ Training
      • Early stopping
      • Learning rate scheduling
      • Model checkpointing
      • Validation monitoring
    
    ✓ Evaluation
      • Accuracy, Precision, Recall, F1
      • ROC-AUC score
      • Confusion matrix
      • Classification report
    
    ✓ Explainability
      • LIME explanations
      • SHAP values
      • Human-readable interpretations
      • Batch explanation generation
    
    ✓ Deployment Ready
      • Social media integration
      • Text/video/image support
      • Batch processing
      • API template included
    """
    print(features)

if __name__ == "__main__":
    import sys
    
    print_banner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['--help', '-h', 'help']:
            print_commands()
        elif command in ['--structure', '-s', 'structure']:
            print_file_structure()
        elif command in ['--quickstart', '-q', 'quickstart']:
            print_quick_start()
        elif command in ['--features', '-f', 'features']:
            print_features()
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable options:")
            print("  python help.py --help          # Show all commands")
            print("  python help.py --structure     # Show file structure")
            print("  python help.py --quickstart    # Show quick start")
            print("  python help.py --features      # Show key features")
    else:
        print("\n🎯 QUICK REFERENCE - Choose an option:\n")
        print("  1. View all commands:     python help.py --help")
        print("  2. File structure:        python help.py --structure")
        print("  3. Quick start:           python help.py --quickstart")
        print("  4. Key features:          python help.py --features")
        print("\n📖 All Documentation Files in Root Directory:")
        print("  • README.md               (Start here)")
        print("  • QUICKSTART.md           (5-minute guide)")
        print("  • INSTALLATION.md         (Detailed setup)")
        print("  • ARCHITECTURE.md         (Technical details)")
        print("  • SOCIAL_MEDIA_INTEGRATION.md  (Deployment)")
        print("  • PROJECT_SUMMARY.md      (Complete overview)")
        print("\n🚀 To start training:")
        print("  python train.py")
        print("\n✅ Setup verified?")
        print("  python setup.py")
        print("\n")

