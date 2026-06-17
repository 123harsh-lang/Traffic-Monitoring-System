"""
AI-Driven Smart Traffic Monitoring System

Configuration settings for paths, model parameters, and thresholds.
"""
import os
from pathlib import Path

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Base project directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data_congestion"
VIDEO_DIR = DATA_DIR / "video"
TEST_VIDEO_DIR = BASE_DIR / "test_videos"

# Label and metadata files
IMAGE_MASTER_FILE = DATA_DIR / "ImageMaster"
INFO_FILE = DATA_DIR / "info.txt"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
FRAMES_DIR = OUTPUT_DIR / "frames"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR, FRAMES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# YOLO CONFIGURATION
# ==============================================================================

# YOLOv8 model variant: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
YOLO_MODEL = "yolov8n"  # Nano model for faster inference

# Vehicle classes to detect (COCO class IDs)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Vehicle class names mapping
VEHICLE_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Detection confidence threshold
DETECTION_CONFIDENCE = 0.25

# NMS IoU threshold
NMS_IOU_THRESHOLD = 0.45

# ==============================================================================
# VIDEO PROCESSING CONFIGURATION
# ==============================================================================

# Skip first frame (corrupted in UCSD dataset)
SKIP_FIRST_FRAME = True

# Frame processing interval (process every Nth frame)
FRAME_INTERVAL = 1

# Maximum frames to process per video (None for all)
MAX_FRAMES = None

# Video display settings
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# ==============================================================================
# CONGESTION CLASSIFICATION CONFIGURATION
# ==============================================================================

# Congestion classes
CONGESTION_CLASSES = ["light", "medium", "heavy"]
NUM_CLASSES = len(CONGESTION_CLASSES)

# Class to index mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CONGESTION_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CONGESTION_CLASSES)}

# Lane configuration
DEFAULT_LANES = 2  # Default number of lanes in one direction

# Congestion thresholds (vehicles PER LANE)
# Updated: 0-2 light, 3-5 medium, 6+ heavy per lane
CONGESTION_THRESHOLDS_PER_LANE = {
    "light": (0, 2),      # 0-2 vehicles per lane = light
    "medium": (3, 5),     # 3-5 vehicles per lane = medium  
    "heavy": (6, float('inf'))  # 6+ vehicles per lane = heavy
}

# Speed-based congestion adjustment (pixels/frame)
# Low speed = worse congestion even with fewer vehicles
SPEED_THRESHOLDS = {
    "stopped": 3,       # < 3 = traffic stopped
    "slow": 10,         # < 10 = slow moving
    "normal": 25        # >= 25 = normal flow
}

# Legacy thresholds (2-lane road)
CONGESTION_THRESHOLDS = {
    "light": (0, 5),
    "medium": (6, 11),
    "heavy": (12, float('inf'))
}

# ==============================================================================
# DASHBOARD CONFIGURATION
# ==============================================================================

# Streamlit page settings
PAGE_TITLE = "Smart Traffic Monitor"
PAGE_ICON = "🚗"
LAYOUT = "wide"

# Color scheme for congestion levels
CONGESTION_COLORS = {
    "light": "#00FF00",    # Green
    "medium": "#FFA500",   # Orange
    "heavy": "#FF0000"     # Red
}

# Accident detection - placeholder for future implementation
ACCIDENT_DETECTION_ENABLED = False

