"""Inference utilities for multimodal sarcasm prediction."""
import os
from typing import Any, Dict, Optional

import torch

from .utils import load_config, get_device, setup_logging
from .model import build_model
from .feature_extraction import MultimodalFeatureExtractor
from .explainability import SarcasmExplainer, simple_explain_prediction

logger = setup_logging(level="INFO")


class MultimodalSarcasmPredictor:
    """Load a trained model and run multimodal predictions with explanations."""

    def __init__(self, config_path: str = "config/config.yaml", model_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.device = get_device(self.config)
        self.model_path = model_path or self._find_latest_model(self.config['paths']['models_dir'])

        self.model = build_model(self.config, self.device)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.feature_extractor = MultimodalFeatureExtractor(self.config, self.device)
        self.explainer = SarcasmExplainer(self.model, self.feature_extractor, self.config, self.device)

        logger.info(f"Loaded model: {self.model_path}")

    @staticmethod
    def _find_latest_model(models_dir: str) -> str:
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        model_files = [
            os.path.join(models_dir, file_name)
            for file_name in os.listdir(models_dir)
            if file_name.startswith("best_model") and file_name.endswith(".pth")
        ]
        if not model_files:
            raise FileNotFoundError(
                f"No trained model found in {models_dir}. Run `python train.py` first."
            )

        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[0]

    def predict(
        self,
        text: Optional[str] = None,
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        generate_detailed_explanation: bool = True,
    ) -> Dict[str, Any]:
        if not any([text, video_path, image_path, audio_path]):
            raise ValueError("At least one modality (text/video/image/audio) is required.")

        with torch.no_grad():
            video_feat, text_feat, audio_feat = self.feature_extractor.extract_multimodal_features(
                video_path=video_path,
                text=text,
                start_time=start_time,
                end_time=end_time,
                image_path=image_path,
                audio_path=audio_path,
            )

            outputs = self.model(
                video_feat.unsqueeze(0),
                text_feat.unsqueeze(0),
                audio_feat.unsqueeze(0),
            )
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_label = int(outputs.argmax(dim=1).item())

        simple_reason = simple_explain_prediction(text or "", pred_label)

        if generate_detailed_explanation:
            try:
                explanation = self.explainer.explain_prediction(
                    text=text,
                    video_path=video_path,
                    image_path=image_path,
                    audio_path=audio_path,
                    video_start=start_time,
                    video_end=end_time,
                )
            except Exception as error:
                explanation = {
                    "text": text,
                    "video_path": video_path,
                    "image_path": image_path,
                    "audio_path": audio_path,
                    "predicted_label": int(pred_label),
                    "predicted_probability": float(probs[pred_label]),
                    "prediction_label_name": "SARCASTIC" if pred_label == 1 else "NOT SARCASTIC",
                    "text_explanation": {
                        "method": str(self.config.get('explainability', {}).get('method', 'lime')).upper(),
                        "error": f"Explanation failed: {error}",
                        "feature_importance": {}
                    }
                }
        else:
            explanation = {
                "method": "SIMPLE",
                "summary": simple_reason,
            }

        return {
            "prediction": pred_label,
            "label_name": "SARCASTIC" if pred_label == 1 else "NOT SARCASTIC",
            "confidence": float(probs[pred_label]),
            "probabilities": {
                "not_sarcastic": float(probs[0]),
                "sarcastic": float(probs[1]),
            },
            "modalities_used": {
                "text": bool(text),
                "video": bool(video_path),
                "image": bool(image_path),
                "audio": bool(audio_path or video_path),
            },
            "simple_explanation": simple_reason,
            "detailed_explanation": explanation,
        }
