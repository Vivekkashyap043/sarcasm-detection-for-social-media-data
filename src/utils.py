"""
Utility functions for the sarcasm detection framework
"""
import os
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
from datetime import datetime

# Prefer loguru if available, otherwise fall back to standard logging
try:
    from loguru import logger as loguru_logger
    _LOGURU_AVAILABLE = True
except Exception:
    loguru_logger = None
    _LOGURU_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file"""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup logging configuration"""
    # Use standard LogRecord attributes for formatting
    log_format = "{levelname} | {name}:{funcName}:{lineno} - {message}"

    if _LOGURU_AVAILABLE and loguru_logger is not None:
        try:
            loguru_logger.remove()
            loguru_logger.add(
                lambda msg: print(msg, end=''),
                format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=level,
                colorize=True
            )
            if log_file:
                os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
                loguru_logger.add(log_file, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level=level, rotation="500 MB")
            return loguru_logger
        except Exception:
            pass

    # Fallback to standard logging
    logger = logging.getLogger('sarcasm')
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        # Use '{' style formatting
        formatter = logging.Formatter(log_format, style='{')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def create_directories(paths: Dict[str, str]):
    """Create necessary directories"""
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get device (CPU or GPU)"""
    device_type = config.get('device', {}).get('device_type', 'cpu')
    # Always use CPU as per configuration
    return torch.device('cpu')


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to readable time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def ensure_dir_exists(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

