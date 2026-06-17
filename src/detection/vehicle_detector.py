"""
Vehicle Detection Module using YOLOv8

This module provides vehicle detection capabilities using a pretrained YOLOv8 model.
It detects cars, motorcycles, buses, and trucks in video frames.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not installed. Run: pip install ultralytics")

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src import config


class VehicleDetector:
    """
    YOLO-based vehicle detector.
    
    Detects vehicles (cars, motorcycles, buses, trucks) in images/frames
    using a pretrained YOLOv8 model.
    """
    
    def __init__(
        self,
        model_name: str = None,
        confidence: float = None,
        iou_threshold: float = None,
        device: str = "auto"
    ):
        """
        Initialize the vehicle detector.
        
        Args:
            model_name: YOLO model variant (e.g., 'yolov8n', 'yolov8s')
            confidence: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name or config.YOLO_MODEL
        self.confidence = confidence or config.DETECTION_CONFIDENCE
        self.iou_threshold = iou_threshold or config.NMS_IOU_THRESHOLD
        self.device = device
        
        self.vehicle_classes = config.VEHICLE_CLASSES
        self.vehicle_names = config.VEHICLE_NAMES
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        if YOLO is None:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model = YOLO(f"{self.model_name}.pt")
        print(f"Loaded YOLO model: {self.model_name}")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Dictionary containing:
                - 'detections': List of detection dictionaries
                - 'vehicle_count': Total number of vehicles
                - 'vehicle_types': Count per vehicle type
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        detections = []
        vehicle_types = {name: 0 for name in self.vehicle_names.values()}
        
        # Process results
        boxes = results.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Only process vehicle classes
                if class_id in self.vehicle_classes:
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    class_name = self.vehicle_names.get(class_id, "vehicle")
                    
                    detection = {
                        'bbox': bbox.tolist(),
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': conf
                    }
                    detections.append(detection)
                    vehicle_types[class_name] += 1
        
        return {
            'detections': detections,
            'vehicle_count': len(detections),
            'vehicle_types': vehicle_types
        }
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect vehicles in multiple frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection results for each frame
        """
        return [self.detect(frame) for frame in frames]
    
    def get_vehicle_density(
        self,
        detection_result: Dict,
        frame_area: int
    ) -> float:
        """
        Calculate vehicle density as ratio of vehicle count to frame area.
        
        Args:
            detection_result: Result from detect()
            frame_area: Frame area in pixels
            
        Returns:
            Vehicle density value
        """
        count = detection_result.get('vehicle_count', 0)
        # Normalize by frame area (scaled for reasonable values)
        return (count / frame_area) * 100000


if __name__ == "__main__":
    # Quick test
    import cv2
    
    print("Testing Vehicle Detector...")
    detector = VehicleDetector()
    
    # Create a dummy frame for testing
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(test_frame)
    
    print(f"Detection result: {result}")
    print("Vehicle Detector test completed!")
