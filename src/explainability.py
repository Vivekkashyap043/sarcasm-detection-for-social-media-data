"""
Explainability module for sarcasm detection predictions
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
try:
    import lime
    import lime.lime_text
    _LIME_AVAILABLE = True
except Exception:
    lime = None
    _LIME_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    shap = None
    _SHAP_AVAILABLE = False
import json
try:
    from .utils import setup_logging
except ImportError:
    from utils import setup_logging

logger = setup_logging(level="INFO")


class SarcasmExplainer:
    """Explain sarcasm detection predictions"""
    
    def __init__(
        self,
        model: nn.Module,
        feature_extractor,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config
        self.device = device
        self.method = config['explainability']['method']
    
    def explain_prediction(
        self,
        text: str = None,
        video_path: str = None,
        image_path: str = None,
        audio_path: str = None,
        video_start: float = 0.0,
        video_end: float = None,
        ground_truth_label: int = None
    ) -> Dict:
        """
        Explain a single prediction for any combination of modalities
        Returns explanation dictionary with feature importance for text, video, audio, and fusion
        """
        logger.info(f"Generating explanation using {self.method}...")
        
        # Get prediction
        pred_label, pred_prob = self._predict(
            text=text,
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
            video_start=video_start,
            video_end=video_end
        )
        
        explanation = {
            'text': text,
            'video_path': video_path,
            'image_path': image_path,
            'audio_path': audio_path,
            'predicted_label': int(pred_label),
            'predicted_probability': float(pred_prob),
            'prediction_label_name': 'SARCASTIC' if pred_label else 'NOT SARCASTIC'
        }
        
        if ground_truth_label is not None:
            explanation['ground_truth_label'] = int(ground_truth_label)
            explanation['ground_truth_label_name'] = 'SARCASTIC' if ground_truth_label else 'NOT SARCASTIC'
            explanation['is_correct'] = pred_label == ground_truth_label
        
        # Text explainability
        if text:
            if self.method.lower() == 'lime':
                text_explanation = self._explain_with_lime(text)
                explanation['text_explanation'] = text_explanation
            elif self.method.lower() == 'shap':
                text_explanation = self._explain_with_shap(text)
                explanation['text_explanation'] = text_explanation
            else:
                logger.warning(f"Explainability method {self.method} not implemented for text")
        
        # Video explainability
        if video_path:
            video_explanation = self._explain_video(video_path, video_start, video_end)
            explanation['video_explanation'] = video_explanation
        
        # Audio explainability
        if audio_path:
            audio_explanation = self._explain_audio(audio_path)
            explanation['audio_explanation'] = audio_explanation
        
        # Multimodal fusion explainability
        visual_path = video_path if video_path else image_path

        if text and visual_path:
            fusion_explanation = self._explain_fusion(text, visual_path, video_start, video_end)
            explanation['fusion_explanation'] = fusion_explanation
        elif text and audio_path:
            fusion_explanation = self._explain_fusion(text, None, None, None, audio_path)
            explanation['fusion_explanation'] = fusion_explanation
        elif visual_path and audio_path:
            fusion_explanation = self._explain_fusion(None, visual_path, video_start, video_end, audio_path)
            explanation['fusion_explanation'] = fusion_explanation
        elif text and visual_path and audio_path:
            fusion_explanation = self._explain_fusion(text, visual_path, video_start, video_end, audio_path)
            explanation['fusion_explanation'] = fusion_explanation
        
        return explanation

    def _explain_video(self, video_path: str, video_start: float = 0.0, video_end: float = None) -> Dict:
        """
        Explain prediction using video features (visual cues)
        """
        try:
            # Extract video features
            frames = self.feature_extractor.video_extractor.extract_frames(video_path, video_start, video_end)
            features = self.feature_extractor.video_extractor.extract_features(frames)
            # Use mean feature importance (dummy example)
            importance = features.mean(dim=0).cpu().numpy().tolist()
            return {
                'method': 'VideoFeatureMean',
                'feature_importance': importance,
                'visual_cues': 'Visual cues (e.g., facial expressions, gestures) may contribute to sarcasm.'
            }
        except Exception as e:
            logger.error(f"Error in video explainability: {str(e)}")
            return {'method': 'VideoFeatureMean', 'error': str(e), 'feature_importance': []}

    def _explain_audio(self, audio_path: str) -> Dict:
        """
        Explain prediction using audio features (prosody, tone)
        """
        try:
            # Dummy audio feature extraction (replace with real extractor)
            # Example: prosody, pitch, energy, etc.
            # Here, just return placeholder
            return {
                'method': 'AudioFeatureDummy',
                'feature_importance': {},
                'audio_cues': 'Audio cues (e.g., tone, pitch, prosody) may contribute to sarcasm.'
            }
        except Exception as e:
            logger.error(f"Error in audio explainability: {str(e)}")
            return {'method': 'AudioFeatureDummy', 'error': str(e), 'feature_importance': {}}

    def _explain_fusion(self, text: str = None, video_path: str = None, video_start: float = 0.0, video_end: float = None, audio_path: str = None) -> Dict:
        """
        Explain prediction using fusion of modalities
        """
        try:
            fusion_info = {}
            if text:
                fusion_info['text'] = 'Text cues contribute to sarcasm (contradiction, exaggeration, etc.)'
            if video_path:
                fusion_info['video'] = 'Visual cues contribute (facial expressions, gestures)'
            if audio_path:
                fusion_info['audio'] = 'Audio cues contribute (tone, prosody)'
            fusion_info['fusion'] = 'Combined cues from modalities enhance sarcasm detection.'
            return fusion_info
        except Exception as e:
            logger.error(f"Error in fusion explainability: {str(e)}")
            return {'fusion': 'Error in fusion explainability', 'error': str(e)}
    
    def _predict(
        self,
        text: str,
        video_path: str = None,
        image_path: str = None,
        audio_path: str = None,
        video_start: float = 0.0,
        video_end: float = None
    ) -> Tuple[int, float]:
        """Make prediction for a sample"""
        self.model.eval()
        
        with torch.no_grad():
            # Extract features
            video_feat, text_feat, audio_feat = self.feature_extractor.extract_multimodal_features(
                video_path=video_path,
                text=text,
                start_time=video_start,
                end_time=video_end,
                image_path=image_path,
                audio_path=audio_path
            )
            
            # Add batch dimension
            video_feat = video_feat.unsqueeze(0)
            text_feat = text_feat.unsqueeze(0)
            audio_feat = audio_feat.unsqueeze(0)
            
            # Forward pass
            outputs = self.model(video_feat, text_feat, audio_feat)
            probs = torch.softmax(outputs, dim=1)
            
            pred_label = outputs.argmax(dim=1).item()
            pred_prob = probs[0, pred_label].item()
        
        return pred_label, pred_prob
    
    def _explain_with_lime(self, text: str) -> Dict:
        """
        Explain prediction using LIME (Local Interpretable Model-agnostic Explanations)
        Focuses on text-based explanation
        """
        try:
            if not _LIME_AVAILABLE:
                return {'method': 'LIME', 'error': 'LIME not installed', 'feature_importance': {}}

            max_chars = int(self.config.get('explainability', {}).get('max_text_chars_for_lime', 1200))
            if len(text or "") > max_chars:
                return {
                    'method': 'LIME',
                    'error': f'Skipped LIME for long text (>{max_chars} chars) for runtime safety',
                    'feature_importance': {}
                }

            # Create LIME explainer
            explainer = lime.lime_text.LimeTextExplainer(
                class_names=['Not Sarcastic', 'Sarcastic'],
                verbose=False
            )
            
            # Create prediction function that LIME can use
            def model_predict_fn(text_inputs):
                """Wrapper function for LIME"""
                predictions = []
                for text in text_inputs:
                    # Extract text features
                    text_feat = self.feature_extractor.text_extractor.extract_features(text)
                    # Create dummy video features
                    video_feat = torch.zeros(1, self.config['features']['video_feature_dim'], device=self.device)
                    
                    # Predict
                    with torch.no_grad():
                        text_feat_batch = text_feat.unsqueeze(0)
                        outputs = self.model(video_feat, text_feat_batch)
                        probs = torch.softmax(outputs, dim=1)
                    
                    predictions.append(probs.cpu().numpy()[0])
                
                return np.array(predictions)
            
            # Generate LIME explanation
            exp = explainer.explain_instance(
                text,
                model_predict_fn,
                num_features=self.config['explainability'].get('num_features', 10),
                num_samples=self.config['explainability'].get('num_samples', 1000)
            )
            
            # Extract feature contributions
            feature_importance = {}
            for word, weight in exp.as_list():
                feature_importance[word] = float(weight)

            # LIME API compatibility across versions
            probabilities = None
            for attr_name in ['predicted_proba', 'predict_proba', 'class_probabilities']:
                attr_val = getattr(exp, attr_name, None)
                if attr_val is not None:
                    probabilities = np.array(attr_val, dtype=np.float32).reshape(-1)
                    break

            if probabilities is None or probabilities.size < 2:
                fallback_probs = model_predict_fn([text])[0]
                probabilities = np.array(fallback_probs, dtype=np.float32).reshape(-1)

            # Guarantee binary shape
            if probabilities.size == 1:
                p1 = float(probabilities[0])
                probabilities = np.array([1.0 - p1, p1], dtype=np.float32)

            predicted_class = int(np.argmax(probabilities))
            
            return {
                'method': 'LIME',
                'feature_importance': feature_importance,
                'predicted_class': predicted_class,
                'class_probabilities': {
                    'not_sarcastic': float(probabilities[0]),
                    'sarcastic': float(probabilities[1])
                }
            }
        
        except Exception as e:
            logger.error(f"Error in LIME explanation: {str(e)}")
            return {
                'method': 'LIME',
                'error': str(e),
                'feature_importance': {}
            }
    
    def _explain_with_shap(self, text: str) -> Dict:
        """
        Explain prediction using SHAP (SHapley Additive exPlanations)
        """
        try:
            if not _SHAP_AVAILABLE:
                return {'method': 'SHAP', 'error': 'SHAP not installed', 'feature_importance': {}}

            # SHAP explanation focuses on important words
            words = text.split()
            
            # Create masker
            masker = shap.maskers.Text(None)
            
            # Create prediction function
            def model_predict_fn(text_inputs):
                """Wrapper function for SHAP"""
                predictions = []
                for text in text_inputs:
                    if isinstance(text, list):
                        text = ' '.join(text)
                    
                    text_feat = self.feature_extractor.text_extractor.extract_features(text)
                    video_feat = torch.zeros(1, self.config['features']['video_feature_dim'], device=self.device)
                    
                    with torch.no_grad():
                        text_feat_batch = text_feat.unsqueeze(0)
                        outputs = self.model(video_feat, text_feat_batch)
                        probs = torch.softmax(outputs, dim=1)
                    
                    predictions.append(probs.cpu().numpy()[0])
                
                return np.array(predictions)
            
            # Create SHAP explainer
            explainer = shap.Explainer(model_predict_fn, masker)
            
            # Generate explanation
            shap_values = explainer([text])
            
            # Extract feature importance
            feature_importance = {}
            for i, word in enumerate(words):
                if i < len(shap_values.values[0]):
                    # Use absolute SHAP values for importance
                    feature_importance[word] = float(np.abs(shap_values.values[0][i]).max())
            
            return {
                'method': 'SHAP',
                'feature_importance': feature_importance,
                'words': words
            }
        
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}")
            return {
                'method': 'SHAP',
                'error': str(e),
                'feature_importance': {}
            }
    
    def generate_explanations_batch(
        self,
        test_data: pd.DataFrame,
        output_dir: str,
        num_samples: int = 20
    ):
        """
        Generate explanations for multiple test samples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Generating {num_samples} explanations...")
        
        explanations = []
        
        # Sample test data
        sample_indices = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
        
        for idx in sample_indices:
            try:
                row = test_data.iloc[idx]
                
                # Determine video path
                video_type = row['video_type']
                video_base = row['video_base']
                
                if video_type == 'c':
                    video_path = os.path.join(
                        self.config['data']['raw_data_path'],
                        'context_videos',
                        f"{video_base}.mp4"
                    )
                else:
                    video_path = os.path.join(
                        self.config['data']['raw_data_path'],
                        'utterance_videos',
                        f"{video_base}.mp4"
                    )
                
                # Generate explanation
                explanation = self.explain_prediction(
                    text=row['SENTENCE'],
                    video_path=video_path,
                    video_start=0.0,
                    video_end=row.get('end_time_seconds', None),
                    ground_truth_label=int(row['Sarcasm'])
                )
                
                explanations.append(explanation)
                
            except Exception as e:
                logger.warning(f"Error generating explanation for sample {idx}: {str(e)}")
        
        # Save explanations
        explanations_file = os.path.join(output_dir, 'explanations.json')
        with open(explanations_file, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        logger.info(f"Explanations saved to {explanations_file}")
        
        # Generate summary
        self._generate_summary(explanations, output_dir)
        
        return explanations
    
    def _generate_summary(self, explanations: List[Dict], output_dir: str):
        """Generate summary of explanations"""
        summary_file = os.path.join(output_dir, 'explanations_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SARCASM DETECTION - EXPLANATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for i, exp in enumerate(explanations):
                f.write(f"Sample {i + 1}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Text: {exp['text'][:100]}...\n\n")
                f.write(f"Prediction: {exp['prediction_label_name']} (confidence: {exp['predicted_probability']:.4f})\n")
                
                if 'ground_truth_label_name' in exp:
                    f.write(f"Ground Truth: {exp['ground_truth_label_name']}\n")
                    f.write(f"Correct: {exp['is_correct']}\n")
                f.write("\n")
                
                # Print top contributing words
                if 'text_explanation' in exp:
                    text_exp = exp['text_explanation']
                    if 'feature_importance' in text_exp:
                        f.write("Top Contributing Words:\n")
                        # Sort by absolute value
                        sorted_features = sorted(
                            text_exp['feature_importance'].items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )
                        for word, importance in sorted_features[:5]:
                            f.write(f"  {word:20} : {importance:8.4f}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        logger.info(f"Summary saved to {summary_file}")


def simple_explain_prediction(text: str, is_sarcastic: int) -> str:
    """
    Generate simple, human-readable explanation for why a prediction is sarcastic
    This is useful for end users
    """
    
    sarcasm_indicators = {
        'way': 'Obvious contradiction patterns detected',
        'sure': 'Overly polite/certain language often indicates sarcasm',
        'yeah': 'Dismissive agreement pattern',
        'right': 'Sarcastic affirmation detected',
        'great': 'Praise word often used sarcastically',
        'love': 'Exaggerated positive sentiment',
        'perfect': 'Over-the-top positive language',
        'obviously': 'Obvious exaggeration indicator'
    }
    
    words_lower = text.lower().split()
    found_indicators = [word for word in words_lower if word in sarcasm_indicators]
    
    if is_sarcastic:
        explanation = "This statement is SARCASTIC because:\n"
        
        if found_indicators:
            explanation += f"1. Contains sarcasm indicators: {', '.join(set(found_indicators))}\n"
        else:
            explanation += "1. The statement contains contradictory or exaggerated language patterns\n"
        
        explanation += "2. The tone appears to convey opposite meaning from literal words\n"
        explanation += "3. Context suggests ironic or mocking intention\n"
        
    else:
        explanation = "This statement appears to be GENUINE (not sarcastic) because:\n"
        explanation += "1. Language is direct and straightforward\n"
        explanation += "2. No contradictory patterns detected\n"
        explanation += "3. Tone matches the literal meaning of the words\n"
    
    return explanation

