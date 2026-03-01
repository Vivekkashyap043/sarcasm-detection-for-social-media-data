"""
Multimodal model architectures for sarcasm detection
"""
import torch
import torch.nn as nn
from typing import Dict


class MultimodalLSTMModel(nn.Module):
    """LSTM-based multimodal architecture"""
    
    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_audio = audio_dim > 0
        
        # Feature projection layers
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.use_audio:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # LSTM layers for each modality
        self.video_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.text_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        if self.use_audio:
            self.audio_lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )

        fusion_input_dim = hidden_dim * (6 if self.use_audio else 4)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        video_features: (batch_size, video_dim)
        text_features: (batch_size, text_dim)
        """
        # Project features
        video_proj = self.video_proj(video_features)  # (batch_size, hidden_dim)
        text_proj = self.text_proj(text_features)  # (batch_size, hidden_dim)
        
        # Add sequence dimension for LSTM
        video_proj = video_proj.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        text_proj = text_proj.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # LSTM encoding
        video_lstm_out, _ = self.video_lstm(video_proj)  # (batch_size, 1, hidden_dim*2)
        text_lstm_out, _ = self.text_lstm(text_proj)  # (batch_size, 1, hidden_dim*2)
        
        # Remove sequence dimension
        video_lstm_out = video_lstm_out.squeeze(1)  # (batch_size, hidden_dim*2)
        text_lstm_out = text_lstm_out.squeeze(1)  # (batch_size, hidden_dim*2)
        
        if self.use_audio:
            if audio_features is None:
                audio_features = torch.zeros(
                    (video_features.size(0), self.audio_dim),
                    dtype=video_features.dtype,
                    device=video_features.device
                )
            audio_proj = self.audio_proj(audio_features).unsqueeze(1)
            audio_lstm_out, _ = self.audio_lstm(audio_proj)
            audio_lstm_out = audio_lstm_out.squeeze(1)
            fused = torch.cat([video_lstm_out, text_lstm_out, audio_lstm_out], dim=1)
        else:
            fused = torch.cat([video_lstm_out, text_lstm_out], dim=1)

        fused = self.fusion(fused)  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(fused)  # (batch_size, num_classes)
        
        return logits


class MultimodalTransformerModel(nn.Module):
    """Transformer-based multimodal architecture"""
    
    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_heads: int = 8
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.use_audio = audio_dim > 0
        
        # Feature projection
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        if self.use_audio:
            self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Positional encoding
        self.n_modalities = 3 if self.use_audio else 2
        self.pos_enc = nn.Embedding(self.n_modalities, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.n_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        video_features: (batch_size, video_dim)
        text_features: (batch_size, text_dim)
        """
        batch_size = video_features.shape[0]
        
        # Project features
        video_proj = self.video_proj(video_features)  # (batch_size, hidden_dim)
        text_proj = self.text_proj(text_features)  # (batch_size, hidden_dim)
        
        modality_features = [video_proj, text_proj]
        if self.use_audio:
            if audio_features is None:
                audio_features = torch.zeros(
                    (video_features.size(0), self.audio_dim),
                    dtype=video_features.dtype,
                    device=video_features.device
                )
            modality_features.append(self.audio_proj(audio_features))

        modalities = torch.stack(modality_features, dim=1)
        
        # Add positional encoding
        pos_ids = torch.arange(self.n_modalities, device=video_features.device).unsqueeze(0).expand(batch_size, -1)
        pos_enc = self.pos_enc(pos_ids)  # (batch_size, 2, hidden_dim)
        modalities = modalities + pos_enc
        
        # Transformer encoding
        encoded = self.transformer_encoder(modalities)  # (batch_size, 2, hidden_dim)
        
        # Flatten and classify
        encoded_flat = encoded.reshape(batch_size, -1)  # (batch_size, hidden_dim*2)
        logits = self.classifier(encoded_flat)  # (batch_size, num_classes)
        
        return logits


class MultimodalMLPModel(nn.Module):
    """Simple MLP-based multimodal architecture"""
    
    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.use_audio = audio_dim > 0
        
        # Build MLP layers
        layers = []
        input_dim = video_dim + text_dim + (audio_dim if self.use_audio else 0)
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        video_features: (batch_size, video_dim)
        text_features: (batch_size, text_dim)
        """
        # Concatenate features
        modalities = [video_features, text_features]
        if self.use_audio:
            if audio_features is None:
                audio_features = torch.zeros(
                    (video_features.size(0), self.audio_dim),
                    dtype=video_features.dtype,
                    device=video_features.device
                )
            modalities.append(audio_features)

        combined = torch.cat(modalities, dim=1)
        
        # MLP processing
        features = self.mlp(combined)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class AttentionMultimodalModel(nn.Module):
    """Attention-based multimodal architecture"""
    
    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.use_audio = audio_dim > 0
        
        # Feature projection
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.use_audio:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion
        fusion_input = hidden_dim * (3 if self.use_audio else 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        video_features: (batch_size, video_dim)
        text_features: (batch_size, text_dim)
        """
        # Project features
        video_proj = self.video_proj(video_features)  # (batch_size, hidden_dim)
        text_proj = self.text_proj(text_features)  # (batch_size, hidden_dim)
        
        # Add sequence dimension
        modality_list = [video_proj.unsqueeze(1), text_proj.unsqueeze(1)]
        if self.use_audio:
            if audio_features is None:
                audio_features = torch.zeros(
                    (video_features.size(0), self.audio_dim),
                    dtype=video_features.dtype,
                    device=video_features.device
                )
            modality_list.append(self.audio_proj(audio_features).unsqueeze(1))

        modalities = torch.cat(modality_list, dim=1)
        
        # Apply attention
        attended, _ = self.attention(modalities, modalities, modalities)  # (batch_size, 2, hidden_dim)
        
        # Mean pooling
        attended_pooled = attended.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Fusion
        fusion_parts = [attended_pooled, video_proj.squeeze(1)]
        if self.use_audio:
            fusion_parts.append(text_proj.squeeze(1))
        fused = torch.cat(fusion_parts, dim=1)
        fused = self.fusion(fused)  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


def build_model(config: Dict, device: torch.device) -> nn.Module:
    """Build model based on configuration"""
    architecture = config['model']['architecture']
    
    model_params = {
        'video_dim': config['features']['video_feature_dim'],
        'text_dim': config['features']['text_feature_dim'],
        'audio_dim': config['features'].get('audio_feature_dim', 0),
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'num_classes': config['model']['output_dim'],
        'dropout': config['model']['dropout']
    }
    
    if architecture == 'multimodal_lstm':
        model = MultimodalLSTMModel(**model_params)
    elif architecture == 'multimodal_transformer':
        model = MultimodalTransformerModel(**model_params)
    elif architecture == 'multimodal_mlp':
        model = MultimodalMLPModel(**model_params)
    elif architecture == 'multimodal_attention':
        model = AttentionMultimodalModel(**model_params)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model.to(device)

