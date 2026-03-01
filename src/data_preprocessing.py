"""
Data preprocessing and loading module for sarcasm detection framework
"""
import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from utils import setup_logging

logger = setup_logging(level="INFO")


class DataPreprocessor:
    """Handle data loading, cleaning, and splitting"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.raw_data_path = config['data']['raw_data_path']
        self.processed_data_path = config['data']['processed_data_path']
        self.train_ratio = config['data']['train_ratio']
        self.test_ratio = config['data']['test_ratio']
        self.random_seed = config['data']['random_seed']
        self.metadata_filename = config['data'].get('metadata_filename', 'metadata.xlsx')
        self.use_full_data_for_training = bool(config['data'].get('use_full_data_for_training', False))
        
        # Ensure directories exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(f"{self.processed_data_path}/train", exist_ok=True)
        os.makedirs(f"{self.processed_data_path}/test", exist_ok=True)
        self._malformed_key_count = 0
    
    def load_metadata(self) -> pd.DataFrame:
        """Load and preprocess metadata from Excel or CSV"""
        preferred_path = os.path.join(self.raw_data_path, self.metadata_filename)
        candidates = [
            preferred_path,
            os.path.join(self.raw_data_path, 'metadata.xlsx'),
            os.path.join(self.raw_data_path, 'metadata.csv')
        ]

        metadata_path = None
        for path in candidates:
            if os.path.exists(path):
                metadata_path = path
                break

        if metadata_path is None:
            raise FileNotFoundError(
                f"No metadata file found. Tried: {', '.join(candidates)}"
            )

        logger.info(f"Loading metadata from {metadata_path}")
        if metadata_path.lower().endswith('.xlsx'):
            df = pd.read_excel(metadata_path)
        else:
            df = pd.read_csv(metadata_path)

        df = self._normalize_columns(df)
        
        logger.info(f"Original dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize metadata columns to canonical names used by the pipeline"""
        original_columns = list(df.columns)
        normalized_map = {c: str(c).strip() for c in df.columns}
        df = df.rename(columns=normalized_map)

        def _find_column(candidates: List[str], required: bool = False) -> str:
            lowered = {str(col).strip().lower(): col for col in df.columns}
            for cand in candidates:
                key = cand.strip().lower()
                if key in lowered:
                    return lowered[key]
            if required:
                raise ValueError(
                    f"Required metadata column not found. Tried {candidates}. "
                    f"Available columns: {list(df.columns)}"
                )
            return ""

        key_col = _find_column(['KEY', 'Key', 'key'], required=True)
        scene_col = _find_column(['SCENE', 'Scene', 'scene'], required=False)
        sentence_col = _find_column(['SENTENCE', 'Sentence', 'sentence', 'utterance', 'text'], required=True)
        end_time_col = _find_column(['END_TIME', 'End_Time', 'end_time', 'END TIME'], required=False)
        sarcasm_col = _find_column(['Sarcasm', 'SARCASM', 'sarcasm', 'label', 'LABEL'], required=True)

        rename_to_canonical = {
            key_col: 'KEY',
            sentence_col: 'SENTENCE',
            sarcasm_col: 'Sarcasm'
        }
        if scene_col:
            rename_to_canonical[scene_col] = 'SCENE'
        if end_time_col:
            rename_to_canonical[end_time_col] = 'END_TIME'

        df = df.rename(columns=rename_to_canonical)

        if 'SCENE' not in df.columns:
            df['SCENE'] = df['KEY'].astype(str).str.extract(r'^(.*)_[cu](?:_\d+)?$')[0].fillna('')
        if 'END_TIME' not in df.columns:
            df['END_TIME'] = ''

        logger.info(f"Original columns: {original_columns}")
        logger.info(f"Normalized columns: {df.columns.tolist()}")
        return df
    
    def clean_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess metadata"""
        # Keep rows with valid labels
        df_clean = df[df['Sarcasm'].notna()].copy()
        
        # Convert Sarcasm to binary (0 or 1)
        sarcasm_series = df_clean['Sarcasm']
        if sarcasm_series.dtype == object:
            mapped = sarcasm_series.astype(str).str.strip().str.lower().map({
                '1': 1,
                '0': 0,
                'true': 1,
                'false': 0,
                'sarcastic': 1,
                'not sarcastic': 0,
                'yes': 1,
                'no': 0
            })
            df_clean['Sarcasm'] = mapped.fillna(0).astype(int)
        else:
            df_clean['Sarcasm'] = sarcasm_series.astype(int)
        
        # Optional columns can be missing in different MUSTARD++ exports
        for optional_text_col, fill_value in [
            ('SPEAKER', 'UNKNOWN'),
            ('Implicit_Emotion', 'NONE'),
            ('Explicit_Emotion', 'NONE')
        ]:
            if optional_text_col in df_clean.columns:
                df_clean[optional_text_col] = df_clean[optional_text_col].fillna(fill_value)

        for optional_num_col in ['Valence', 'Arousal']:
            if optional_num_col in df_clean.columns:
                df_clean[optional_num_col] = pd.to_numeric(df_clean[optional_num_col], errors='coerce')
                df_clean[optional_num_col] = df_clean[optional_num_col].fillna(df_clean[optional_num_col].median())
        
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        logger.info(f"Sarcasm distribution:\n{df_clean['Sarcasm'].value_counts()}")
        
        return df_clean
    
    def extract_video_info(self, key: str, scene: str = None, sarcasm: float = None) -> Tuple[str, str, int]:
        """
        Extract video base name and segment index from KEY
        Example: 1_10004_c_00 -> (1_10004_c, c, 0)
        """
        key_str = str(key).strip() if not pd.isna(key) else ""
        scene_str = str(scene).strip() if scene is not None and not pd.isna(scene) else ""

        # Pattern with segment index, e.g. 1_10004_c_00 or 1_S09E01_027_c_5
        match_with_segment = re.match(r'^(.*_[cu])_(\d+)$', key_str)
        if match_with_segment:
            video_base = match_with_segment.group(1)
            segment_idx = int(match_with_segment.group(2))
            video_type = 'c' if video_base.endswith('_c') else 'u'
            return video_base, video_type, segment_idx

        # Pattern without explicit segment index, e.g. 1_10004_u
        match_without_segment = re.match(r'^(.*_[cu])$', key_str)
        if match_without_segment:
            video_base = match_without_segment.group(1)
            video_type = 'c' if video_base.endswith('_c') else 'u'
            return video_base, video_type, 0

        # Fallback for malformed KEY rows (e.g. KEY="Disgust").
        # Sarcasm rows are utterance rows in this dataset.
        fallback_type = 'u' if not pd.isna(sarcasm) else 'c'
        if scene_str:
            self._malformed_key_count += 1
            if self._malformed_key_count <= 5:
                logger.warning(
                    f"Malformed KEY '{key_str}' for SCENE '{scene_str}'. "
                    f"Using fallback video base '{scene_str}_{fallback_type}'."
                )
            return f"{scene_str}_{fallback_type}", fallback_type, 0

        # Last resort (should be very rare)
        self._malformed_key_count += 1
        if self._malformed_key_count <= 5:
            logger.warning(f"Malformed KEY '{key_str}' with missing SCENE. Using zero defaults.")
        return "", fallback_type, 0
    
    def convert_timestamp_to_seconds(self, time_str: str) -> float:
        """Convert timestamp string to seconds"""
        if pd.isna(time_str) or time_str == '':
            return 0.0

        time_str = str(time_str).strip()
        parts = time_str.split(':')

        try:
            if len(parts) == 1:  # ss(.ms) format
                return float(parts[0])
            if len(parts) == 2:  # m:ss(.ms) format
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            if len(parts) == 3:  # h:mm:ss(.ms) format
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except (TypeError, ValueError):
            logger.warning(f"Invalid timestamp format encountered: {time_str}")

        return 0.0
    
    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets (stratified by Sarcasm label)"""

        if self.use_full_data_for_training:
            logger.info("use_full_data_for_training=True -> using 100% data for training")
            empty_test = pd.DataFrame(columns=df.columns)
            return df.reset_index(drop=True), empty_test
        
        logger.info(f"Splitting data: {self.train_ratio*100}% train, {self.test_ratio*100}% test")
        
        # Stratified split to maintain sarcasm distribution
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            train_size=self.train_ratio,
            stratify=df['Sarcasm'],
            random_state=self.random_seed
        )
        
        logger.info(f"Train set size: {len(train_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        logger.info(f"Train sarcasm distribution:\n{train_df['Sarcasm'].value_counts()}")
        logger.info(f"Test sarcasm distribution:\n{test_df['Sarcasm'].value_counts()}")
        
        return train_df, test_df
    
    def save_split_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save split data to CSV files"""
        train_path = os.path.join(self.processed_data_path, 'train', 'metadata.csv')
        test_path = os.path.join(self.processed_data_path, 'test', 'metadata.csv')
        
        train_df.to_csv(train_path, index=False)
        if test_df is not None:
            test_df.to_csv(test_path, index=False)
        
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
    
    def process_data(self):
        """Main data processing pipeline"""
        logger.info("=" * 50)
        logger.info("Starting Data Preprocessing Pipeline")
        logger.info("=" * 50)
        
        # Load metadata
        df = self.load_metadata()
        
        # Clean metadata
        df_clean = self.clean_metadata(df)
        
        # Split data
        train_df, test_df = self.split_train_test(df_clean)
        
        # Add processed information
        train_video_info = train_df.apply(
            lambda row: self.extract_video_info(row['KEY'], row['SCENE'], row['Sarcasm']),
            axis=1
        )
        train_df['video_base'] = train_video_info.apply(lambda t: t[0])
        train_df['video_type'] = train_video_info.apply(lambda t: t[1])
        train_df['segment_idx'] = train_video_info.apply(lambda t: t[2])
        train_df['end_time_seconds'] = train_df['END_TIME'].apply(self.convert_timestamp_to_seconds)

        if len(test_df) > 0:
            test_video_info = test_df.apply(
                lambda row: self.extract_video_info(row['KEY'], row['SCENE'], row['Sarcasm']),
                axis=1
            )
            test_df['video_base'] = test_video_info.apply(lambda t: t[0])
            test_df['video_type'] = test_video_info.apply(lambda t: t[1])
            test_df['segment_idx'] = test_video_info.apply(lambda t: t[2])
            test_df['end_time_seconds'] = test_df['END_TIME'].apply(self.convert_timestamp_to_seconds)
        else:
            test_df = pd.DataFrame(columns=train_df.columns)
        
        # Save split data
        self.save_split_data(train_df, test_df)
        
        logger.info("=" * 50)
        logger.info("Data Preprocessing Completed Successfully")
        logger.info("=" * 50)
        
        return train_df, test_df


def verify_video_files(data_path: str, metadata_df: pd.DataFrame):
    """Verify that all referenced video files exist"""
    logger.info("Verifying video files...")
    
    context_videos_path = os.path.join(data_path, 'context_videos')
    utterance_videos_path = os.path.join(data_path, 'utterance_videos')
    
    missing_videos = []
    
    for _, row in metadata_df.iterrows():
        video_base = row['video_base']
        video_type = row['video_type']
        
        if video_type == 'c':
            video_path = os.path.join(context_videos_path, f"{video_base}.mp4")
        else:
            video_path = os.path.join(utterance_videos_path, f"{video_base}.mp4")
        
        if not os.path.exists(video_path):
            missing_videos.append((row['KEY'], video_path))
    
    if missing_videos:
        logger.warning(f"Found {len(missing_videos)} missing video files:")
        for key, path in missing_videos[:10]:  # Show first 10
            logger.warning(f"  KEY: {key}, Path: {path}")
    else:
        logger.info("All video files verified successfully!")
    
    return missing_videos

