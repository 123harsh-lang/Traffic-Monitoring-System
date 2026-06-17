"""Detection module for vehicle detection and tracking."""
from .vehicle_detector import VehicleDetector
from .video_processor import VideoProcessor
from .motion_tracker import MotionTracker

__all__ = ['VehicleDetector', 'VideoProcessor', 'MotionTracker']
