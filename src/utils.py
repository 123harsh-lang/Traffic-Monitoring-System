"""
Utility functions for logging, file I/O, and visualization.
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from . import config


def setup_logging(name: str = "TrafficMonitor", level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # File handler
    log_file = config.LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on frame.
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries with 'bbox' and 'class' keys
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with bounding boxes drawn
    """
    frame_copy = frame.copy()
    
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            class_name = det.get('class_name', 'vehicle')
            confidence = det.get('confidence', 0)
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame_copy,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
    
    return frame_copy


def add_info_overlay(
    frame: np.ndarray,
    vehicle_count: int,
    congestion_level: str = "N/A",
    fps: float = 0.0
) -> np.ndarray:
    """
    Add information overlay to frame.
    
    Args:
        frame: Input frame
        vehicle_count: Number of vehicles detected
        congestion_level: Current congestion level
        fps: Frames per second
        
    Returns:
        Frame with info overlay
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
    
    # Add text
    cv2.putText(frame_copy, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Congestion level with color
    color_hex = config.CONGESTION_COLORS.get(congestion_level.lower(), "#FFFFFF")
    # Convert hex to BGR
    color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
    cv2.putText(frame_copy, f"Congestion: {congestion_level}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    
    cv2.putText(frame_copy, f"FPS: {fps:.1f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_copy


def get_video_info(video_path: Path) -> Dict:
    """
    Get video file information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video info
    """
    cap = cv2.VideoCapture(str(video_path))
    
    info = {
        'path': str(video_path),
        'name': video_path.stem,
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    }
    
    cap.release()
    return info


def resize_frame(frame: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        width: Target width (optional)
        height: Target height (optional)
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    
    if width is None:
        ratio = height / h
        new_size = (int(w * ratio), height)
    elif height is None:
        ratio = width / w
        new_size = (width, int(h * ratio))
    else:
        new_size = (width, height)
    
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
