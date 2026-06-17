"""
Feature Extraction Module

This module was designed to extract features for ML-based classification.
Currently not used since we use rule-based congestion classification.

Features that can be extracted:
- Vehicle count statistics (mean, max, std)
- Vehicle density
- Vehicle type ratios
- Temporal features (hour of day)

To use:
1. Implement extract_from_video() to process detection results
2. Use prepare_training_data() to create X, y arrays
3. Train classifier with the features
"""

# TODO: Implement when training a congestion classifier
