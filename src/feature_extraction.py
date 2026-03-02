"""Feature extraction module for text, image/video and audio modalities."""
import os
import warnings
import cv2
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from PIL import Image
from torch import nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    _LIBROSA_AVAILABLE = False

try:
    from .utils import setup_logging
except ImportError:
    from utils import setup_logging

logger = setup_logging(level="INFO")


class VideoFeatureExtractor:
    """Extract visual features from video frames"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.target_fps = config['video']['target_fps']
        self.frame_size = config['video']['frame_size']
        self.n_frames_per_segment = config['video']['n_frames_per_segment']
        self.video_feature_dim = int(config['features']['video_feature_dim'])

        # Load pre-trained ResNet18 (CPU-friendly, 512-dim penultimate layer)
        logger.info("Loading pre-trained ResNet18 model...")
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove classification layer
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False
    
    def extract_frames(self, video_path: str, start_time: float = 0.0, end_time: float = None) -> np.ndarray:
        """
        Extract frames from video between start_time and end_time
        Returns: numpy array of shape (n_frames, 3, height, width)
        """
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return np.zeros((self.n_frames_per_segment, 3, self.frame_size[0], self.frame_size[1]))
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Unable to open video: {video_path}")
                return np.zeros((self.n_frames_per_segment, 3, self.frame_size[0], self.frame_size[1]))

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if fps <= 0.0 or total_frames <= 0:
                cap.release()
                logger.warning(
                    f"Invalid video metadata for frame extraction: {video_path} (fps={fps}, frames={total_frames})"
                )
                return np.zeros((self.n_frames_per_segment, 3, self.frame_size[0], self.frame_size[1]))

            if end_time is None:
                end_time = total_frames / fps
            
            # Calculate frame indices
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Ensure valid range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            # Sample n_frames_per_segment frames evenly
            frame_indices = np.linspace(start_frame, end_frame - 1, self.n_frames_per_segment, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Resize frame
                    frame = cv2.resize(frame, tuple(self.frame_size))
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                return np.zeros((self.n_frames_per_segment, 3, self.frame_size[0], self.frame_size[1]))
            
            # Pad if necessary
            while len(frames) < self.n_frames_per_segment:
                frames.append(frames[-1])
            
            # Convert to CHW format (n_frames, channels, height, width)
            frames_array = np.array(frames[:self.n_frames_per_segment])  # (n_frames, height, width, 3)
            frames_array = np.transpose(frames_array, (0, 3, 1, 2))  # (n_frames, 3, height, width)
            return frames_array
        
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return np.zeros((self.n_frames_per_segment, 3, self.frame_size[0], self.frame_size[1]))

    def extract_frames_from_image(self, image_path: str) -> np.ndarray:
        """Convert a single image into repeated frames for visual features"""
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return np.zeros((self.n_frames_per_segment, 3, self.frame_size[0], self.frame_size[1]))

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(tuple(self.frame_size))
            frame = np.array(image)

            # Convert to CHW and repeat to match expected frames
            frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)
            frames = np.repeat(frame[np.newaxis, ...], self.n_frames_per_segment, axis=0)
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames from image {image_path}: {str(e)}")
            return np.zeros((self.n_frames_per_segment, 3, self.frame_size[0], self.frame_size[1]))
    
    def extract_features(self, frames: np.ndarray) -> torch.Tensor:
        """
        Extract features from frames using ResNet50
        Input: (n_frames, 3, height, width)
        Output: (n_frames, feature_dim)
        """
        # Normalize frames
        frames = frames.astype(np.float32) / 255.0
        
        # ImageNet normalization (ensure same dtype)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        frames = (frames - mean) / std
        
        # Convert to tensor and move to device
        frames_tensor = torch.from_numpy(frames).float().to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(frames_tensor)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        
        return features


class TextFeatureExtractor:
    """Extract textual features from sentences"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.model_name = config['text']['model_name']
        self.max_length = config['text']['max_length']
        self.text_feature_dim = config['features']['text_feature_dim']
        
        # Load pre-trained model and tokenizer
        logger.info(f"Loading {self.model_name} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()
        
        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False
    
    def extract_features(self, text: str) -> torch.Tensor:
        """
        Extract features from text using DistilBERT
        Input: text string
        Output: (feature_dim,) tensor - mean pooled representation
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token representation
                features = outputs.last_hidden_state[:, 0, :]  # (1, feature_dim)
            
            return features.squeeze(0)
        
        except Exception as e:
            logger.error(f"Error extracting text features: {str(e)}")
            return torch.zeros(self.text_feature_dim, device=self.device)


class AudioFeatureExtractor:
    """Extract lightweight MFCC-based audio features from audio/video files."""

    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.use_video_audio = bool(config.get('audio', {}).get('use_video_audio', False))
        self.audio_feature_dim = int(config['features'].get('audio_feature_dim', 80))
        self.target_sr = int(config.get('audio', {}).get('sample_rate', 16000))
        self.max_audio_length = float(config.get('audio', {}).get('max_audio_length', 30.0))
        self.n_mfcc = min(40, max(8, self.audio_feature_dim // 2))
        self.cache_waveforms = bool(config.get('audio', {}).get('cache_waveforms', True))
        self.cache_size = int(config.get('audio', {}).get('cache_size', 256))
        self._waveform_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._cache_order: List[str] = []
        self._failed_audio_paths: set[str] = set()

    @staticmethod
    def _is_audio_file(path: str) -> bool:
        audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
        ext = os.path.splitext(path)[1].lower()
        return ext in audio_exts

    @staticmethod
    def _is_video_file(path: str) -> bool:
        video_exts = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}
        ext = os.path.splitext(path)[1].lower()
        return ext in video_exts

    def _get_audio_waveform(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        if self.cache_waveforms and audio_path in self._waveform_cache:
            cached_y, cached_sr = self._waveform_cache[audio_path]
            return cached_y, cached_sr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            y, sr = librosa.load(audio_path, sr=self.target_sr)

        if self.cache_waveforms:
            if audio_path not in self._waveform_cache:
                self._waveform_cache[audio_path] = (y, sr)
                self._cache_order.append(audio_path)
            while len(self._cache_order) > self.cache_size:
                oldest = self._cache_order.pop(0)
                self._waveform_cache.pop(oldest, None)

        return y, sr

    def extract_features(
        self,
        audio_path: Optional[str],
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> torch.Tensor:
        if not audio_path or not os.path.exists(audio_path) or not _LIBROSA_AVAILABLE:
            return torch.zeros(self.audio_feature_dim, device=self.device)

        # If this file previously failed decoding, skip retrying for every comment/post reuse.
        if audio_path in self._failed_audio_paths:
            return torch.zeros(self.audio_feature_dim, device=self.device)

        ext = os.path.splitext(audio_path)[1].lower()
        is_audio = self._is_audio_file(audio_path)
        is_video = self._is_video_file(audio_path)

        # Skip unknown/no-extension media files (common for unresolved social links).
        if not is_audio and not is_video:
            return torch.zeros(self.audio_feature_dim, device=self.device)

        # For training/inference, avoid extracting audio from video containers unless explicitly enabled.
        if not self.use_video_audio and is_video:
            return torch.zeros(self.audio_feature_dim, device=self.device)

        try:
            y_all, sr = self._get_audio_waveform(audio_path)

            if y_all is None or len(y_all) == 0:
                return torch.zeros(self.audio_feature_dim, device=self.device)

            start_sec = max(0.0, float(start_time))
            if end_time is not None and end_time > start_sec:
                end_sec = float(end_time)
            elif self.max_audio_length > 0:
                end_sec = start_sec + self.max_audio_length
            else:
                end_sec = len(y_all) / max(sr, 1)

            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            end_sample = min(end_sample, len(y_all))
            start_sample = min(start_sample, end_sample)
            y = y_all[start_sample:end_sample]

            if y is None or len(y) == 0:
                return torch.zeros(self.audio_feature_dim, device=self.device)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            features = np.concatenate([mfcc_mean, mfcc_std], axis=0)

            if len(features) < self.audio_feature_dim:
                features = np.pad(features, (0, self.audio_feature_dim - len(features)))
            elif len(features) > self.audio_feature_dim:
                features = features[:self.audio_feature_dim]

            return torch.tensor(features, dtype=torch.float32, device=self.device)
        except Exception as e:
            self._failed_audio_paths.add(audio_path)
            message = str(e).strip() if str(e).strip() else e.__class__.__name__
            logger.warning(
                f"Audio feature extraction skipped for {audio_path} (unreadable/unsupported audio stream): {message}"
            )
            return torch.zeros(self.audio_feature_dim, device=self.device)


class MultimodalFeatureExtractor:
    """Combine video/image, text and audio features."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.video_extractor = VideoFeatureExtractor(config, device)
        self.text_extractor = TextFeatureExtractor(config, device)
        self.audio_extractor = AudioFeatureExtractor(config, device)
        self.audio_feature_dim = int(config['features'].get('audio_feature_dim', 80))

    def extract_multimodal_features(
        self,
        video_path: Optional[str],
        text: Optional[str],
        start_time: float = 0.0,
        end_time: float = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Visual features (video preferred, then image, then zeros)
        if video_path:
            frames = self.video_extractor.extract_frames(video_path, start_time, end_time)
            video_features = self.video_extractor.extract_features(frames)
            video_features_pooled = video_features.mean(dim=0)
        elif image_path:
            frames = self.video_extractor.extract_frames_from_image(image_path)
            video_features = self.video_extractor.extract_features(frames)
            video_features_pooled = video_features.mean(dim=0)
        else:
            video_features_pooled = torch.zeros(
                int(self.config['features']['video_feature_dim']),
                device=self.device
            )

        safe_text = text if text is not None else ""
        text_features = self.text_extractor.extract_features(safe_text)

        audio_source = audio_path if audio_path else video_path
        audio_features = self.audio_extractor.extract_features(audio_source, start_time=start_time, end_time=end_time)

        return video_features_pooled, text_features, audio_features
    
    def extract_video_text_features(
        self,
        video_path: Optional[str],
        text: Optional[str],
        start_time: float = 0.0,
        end_time: float = None,
        image_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible two-modality API used by older scripts."""
        video_features, text_features, _ = self.extract_multimodal_features(
            video_path=video_path,
            text=text,
            start_time=start_time,
            end_time=end_time,
            image_path=image_path,
            audio_path=None
        )
        return video_features, text_features
    
    def extract_batch_features(
        self,
        batch_data: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features for a batch of samples
        batch_data: list of dicts with keys: 'video_path', 'text', 'start_time', 'end_time'
        """
        video_features_list = []
        text_features_list = []
        audio_features_list = []
        
        for sample in batch_data:
            video_feat, text_feat, audio_feat = self.extract_multimodal_features(
                video_path=sample.get('video_path'),
                text=sample.get('text'),
                start_time=sample.get('start_time', 0.0),
                end_time=sample.get('end_time', None),
                image_path=sample.get('image_path'),
                audio_path=sample.get('audio_path')
            )
            
            video_features_list.append(video_feat)
            text_features_list.append(text_feat)
            audio_features_list.append(audio_feat)
        
        # Stack into batch
        video_features = torch.stack(video_features_list)
        text_features = torch.stack(text_features_list)
        audio_features = torch.stack(audio_features_list)

        return video_features, text_features, audio_features


# Feature fusion methods
class FeatureFusion(nn.Module):
    """Combine video and text features"""
    
    def __init__(self, video_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Fuse video and text features"""
        video_proj = torch.relu(self.video_proj(video_features))
        text_proj = torch.relu(self.text_proj(text_features))
        
        # Concatenate and fuse
        fused = torch.cat([video_proj, text_proj], dim=-1)
        fused = torch.relu(self.fusion(fused))
        
        return fused

