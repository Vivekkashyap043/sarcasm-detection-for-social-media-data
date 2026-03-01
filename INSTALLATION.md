# Installation and Setup Guide

Complete guide for setting up the Multimodal Sarcasm Detection Framework on your system.

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM (for CPU inference, 16GB+ recommended for training)
- ~10GB disk space for models and data

### Recommended Specifications
- Python 3.10 or 3.11
- 16GB+ RAM
- SSD with 50GB+ space
- CPU: Intel i5/Ryzen 5 or better (for reasonable training speed)
- GPU (optional): NVIDIA with CUDA support for faster training

## Step-by-Step Installation

### 1. Check Python Version

```bash
python --version
# Should be Python 3.8 or higher
```

If you have multiple Python versions, use `python3` instead of `python`:
```bash
python3 --version
```

### 2. Create Virtual Environment

#### Using venv (Built-in)
```bash
# Create virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Using Conda (Alternative)
```bash
conda create -n sarcasm_detection python=3.10
conda activate sarcasm_detection
```

### 3. Install PyTorch

PyTorch installation depends on your system (CPU vs GPU). Choose ONE of the options:

#### Option A: CPU Only (Recommended for Windows without GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Option B: GPU with CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option C: GPU with CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Option D: Using Conda (Any Platform)
```bash
# CPU only
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# With CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 4. Verify PyTorch Installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 5. Install Project Dependencies

```bash
# Navigate to project directory
cd sarcasm-detection

# Install required packages
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### 6. Verify Installation

```bash
python setup.py --skip-verification
```

If there are no errors, your installation is complete!

## Post-Installation Setup

### 1. Prepare Data

Ensure your MUSTARD++ dataset is organized:
```
data/
├── metadata.csv           # Main annotation file
├── context_videos/        # Folder with context videos
└── utterance_videos/      # Folder with utterance videos
```

### 2. Configure Settings

Edit `config/config.yaml` for your specific setup:

```yaml
# For CPU, use smaller batch sizes:
training:
  batch_size: 4  # or 8

# For slower systems:
video:
  target_fps: 1  # Instead of 2

# For faster training (lower accuracy):
model:
  hidden_dim: 256  # Instead of 512
```

### 3. Test Setup

```bash
python setup.py
```

This will:
- Create necessary directories
- Verify dependencies
- Check data structure
- Validate configuration

## Platform-Specific Notes

### Windows

**Common Issue**: `ModuleNotFoundError: No module named 'cv2'`
```bash
# Solution:
pip install opencv-python

# If still failing:
pip install --upgrade opencv-python
```

**Activation on PowerShell**:
```powershell
# Use:
venv\Scripts\Activate.ps1

# If running unsigned script error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS

**For Apple Silicon (M1/M2/M3)**:
```bash
# Use conda (recommended for Apple Silicon):
conda create -n sarcasm_detection python=3.10
conda activate sarcasm_detection

# Install PyTorch for Apple Silicon:
conda install pytorch torchvision torchaudio -c pytorch

# Then install other requirements:
pip install -r requirements.txt
```

**OpenCV on Apple Silicon**:
```bash
pip install opencv-python-headless
```

### Linux

**Ubuntu/Debian**:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

**Fedora/CentOS**:
```bash
sudo dnf install python3-devel

python3 -m venv venv
source venv/bin/activate
```

## Troubleshooting

### Issue: ImportError for PyTorch
**Solution**: Check your installation:
```bash
python -c "import torch; print(torch.__version__)"
```

If failing, reinstall:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: `No module named 'cv2'`
**Solution**:
```bash
pip install --upgrade opencv-python
```

### Issue: Out of Memory (OOM) Error
**Solution**: Reduce batch size in `config/config.yaml`:
```yaml
training:
  batch_size: 2  # Reduce from 8 to 4 or 2
```

### Issue: Video file not found
**Solution**: 
1. Check file paths in `data/` directory
2. Ensure video files are named correctly
3. Verify metadata.csv references correct file names

### Issue: Slow training on CPU
**Solution**: This is normal. Options:
- Use a GPU (if available)
- Use smaller batch sizes and patience
- Use MLP architecture instead of Transformer
- Reduce number of epochs for testing

### Issue: ModuleNotFoundError for specific package
**Solution**: Reinstall all requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Upgrading Packages

To update all packages to latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

To check for outdated packages:
```bash
pip list --outdated
```

## Managing Multiple Projects

If you're working with multiple Python projects:

```bash
# Use environment name in venv
python -m venv venv_sarcasm

# Or use conda with named environments
conda create -n sarcasm_detection python=3.10
conda activate sarcasm_detection
```

## Performance Optimization

### For Faster Training

1. **Use GPU if available**:
   - Install CUDA-enabled PyTorch
   - Increase batch size to 16-32
   - Use Transformer architecture

2. **On CPU, optimize**:
   - Increase batch size to 8-16
   - Use MLP architecture
   - Reduce model hidden dimension to 256
   - Extract fewer video frames (fps=1)

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile your training
python -m memory_profiler train.py
```

### Training on WSL2 (Windows Subsystem for Linux)

```bash
# Install WSL2 and Ubuntu
wsl --install

# Inside WSL2:
sudo apt-get update
sudo apt-get install python3.10 python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Docker Installation (Advanced)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Copy project
COPY . .

CMD ["python", "train.py"]
```

Build and run:
```bash
docker build -t sarcasm-detection .
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models sarcasm-detection
```

## Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] PyTorch installed: `python -c "import torch"`
- [ ] TensorFlow alternatives working: `python -c "import transformers"`
- [ ] OpenCV working: `python -c "import cv2"`
- [ ] Data directory exists with metadata.csv
- [ ] Video files present in appropriate directories
- [ ] config/config.yaml is readable
- [ ] models/ and results/ directories exist
- [ ] Setup script passes: `python setup.py`

## Getting Help

If you encounter issues:

1. Check this guide for your specific issue
2. Review error messages carefully
3. Search online for the specific error
4. Check dependencies are compatible with your Python version
5. Try reinstalling requirements: `pip install --force-reinstall -r requirements.txt`

## Next Steps

Once installation is complete:

1. Read [QUICKSTART.md](QUICKSTART.md) for quick start guide
2. Read [README.md](README.md) for full documentation
3. Run `python train.py` to start training
4. Check [SOCIAL_MEDIA_INTEGRATION.md](SOCIAL_MEDIA_INTEGRATION.md) for deployment

---

**For issues**: Check error logs in `logs/training.log`

