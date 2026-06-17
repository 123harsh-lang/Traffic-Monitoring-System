"""
Model Training Script

This script handles the end-to-end training pipeline:
1. Load and preprocess data
2. Extract features from videos
3. Train congestion classifier
4. Evaluate and save model
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src import config
from src.features import DataLoader, FeatureExtractor
from src.classification.congestion_model import CongestionClassifier


def load_or_extract_features(
    features_path: Path = None,
    max_videos: int = None,
    max_frames: int = 10,
    force_extract: bool = False
) -> pd.DataFrame:
    """
    Load existing features or extract new ones.
    
    Args:
        features_path: Path to cached features CSV
        max_videos: Maximum videos to process
        max_frames: Maximum frames per video
        force_extract: Force re-extraction even if cache exists
        
    Returns:
        DataFrame with features and labels
    """
    features_path = Path(features_path or config.OUTPUT_DIR / "extracted_features.csv")
    
    # Check for cached features
    if features_path.exists() and not force_extract:
        print(f"Loading cached features from: {features_path}")
        return pd.read_csv(features_path)
    
    # Extract features
    print("Extracting features from videos...")
    extractor = FeatureExtractor()
    features_df = extractor.extract_from_dataset(
        max_videos=max_videos,
        max_frames_per_video=max_frames,
        save_path=features_path
    )
    
    return features_df


def prepare_data(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare training and validation data.
    
    Args:
        features_df: DataFrame with features
        test_size: Validation set proportion
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, scaler)
    """
    # Feature columns
    feature_cols = [
        'mean_vehicle_count',
        'max_vehicle_count',
        'std_vehicle_count',
        'vehicle_density',
        'car_ratio',
        'motorcycle_ratio',
        'bus_ratio',
        'truck_ratio'
    ]
    
    # Add time features if available
    if 'hour' in features_df.columns:
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        feature_cols.extend(['hour_sin', 'hour_cos'])
    
    # Extract X and y
    X = features_df[feature_cols].values.astype(np.float32)
    y = features_df['class_idx'].values.astype(np.int32)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    return X_train, X_val, y_train, y_val, scaler


def train_model(
    features_df: pd.DataFrame = None,
    max_videos: int = None,
    epochs: int = None,
    save_model: bool = True
) -> Tuple[CongestionClassifier, dict]:
    """
    Train the congestion classification model.
    
    Args:
        features_df: Pre-loaded features (optional)
        max_videos: Maximum videos to use
        epochs: Training epochs
        save_model: Whether to save trained model
        
    Returns:
        Tuple of (trained classifier, training history)
    """
    # Load or extract features
    if features_df is None:
        features_df = load_or_extract_features(max_videos=max_videos)
    
    print(f"\nDataset size: {len(features_df)} samples")
    print(f"Class distribution:\n{features_df['class'].value_counts()}")
    
    # Prepare data
    X_train, X_val, y_train, y_val, scaler = prepare_data(features_df)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Input dimension: {X_train.shape[1]}")
    
    # Create and train classifier
    classifier = CongestionClassifier(input_dim=X_train.shape[1])
    
    print("\nModel architecture:")
    classifier.summary()
    
    print("\nTraining model...")
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs or config.EPOCHS
    )
    
    # Evaluate
    loss, accuracy = classifier.evaluate(X_val, y_val)
    print(f"\nValidation Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Save model and scaler
    if save_model:
        classifier.save()
        
        scaler_path = config.MODEL_DIR / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {scaler_path}")
    
    return classifier, history


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description='Train Congestion Classifier')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum videos to process')
    parser.add_argument('--max-frames', type=int, default=10,
                        help='Maximum frames per video')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--force-extract', action='store_true',
                        help='Force feature re-extraction')
    parser.add_argument('--validation-only', action='store_true',
                        help='Only validate existing model')
    
    args = parser.parse_args()
    
    if args.validation_only:
        # Load and validate existing model
        print("Loading existing model for validation...")
        classifier = CongestionClassifier(input_dim=10)
        try:
            classifier.load()
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("No saved model found. Train a model first.")
            return
    else:
        # Train new model
        features_df = load_or_extract_features(
            max_videos=args.max_videos,
            max_frames=args.max_frames,
            force_extract=args.force_extract
        )
        
        classifier, history = train_model(
            features_df=features_df,
            epochs=args.epochs
        )
        
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
