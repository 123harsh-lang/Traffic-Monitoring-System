"""
Motion Detection and Speed Estimation Module

Uses optical flow and vehicle tracking to estimate vehicle speeds
and detect stopped/slow-moving vehicles for better congestion detection.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class MotionTracker:
    """
    Track vehicle motion across frames to estimate speed.
    
    Uses centroid tracking to follow vehicles between frames
    and calculate their displacement (proxy for speed).
    """
    
    def __init__(
        self,
        max_disappeared: int = 5,
        max_distance: int = 100,
        history_length: int = 10
    ):
        """
        Initialize motion tracker.
        
        Args:
            max_disappeared: Frames before object is deregistered
            max_distance: Max distance for centroid matching
            history_length: Number of frames to keep for speed calculation
        """
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.disappeared = {}  # object_id -> frames disappeared
        self.history = {}  # object_id -> deque of positions
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_length = history_length
        
        # Previous frame for optical flow
        self.prev_frame = None
        self.prev_gray = None
        
        # Speed statistics
        self.speeds = []
        self.avg_speed = 0.0
    
    def register(self, centroid: Tuple[int, int]):
        """Register a new object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = deque(maxlen=self.history_length)
        self.history[self.next_object_id].append(centroid)
        self.next_object_id += 1
    
    def deregister(self, object_id: int):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]
    
    def update(self, detections: List[Dict]) -> Dict:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox' key
            
        Returns:
            Dict with tracking results and speed info
        """
        # Extract centroids from detections
        input_centroids = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            input_centroids.append((cx, cy))
        
        # If no detections, mark all as disappeared
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._get_results()
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self._get_results()
        
        # Match existing objects to new centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        # Calculate distances
        D = np.zeros((len(object_centroids), len(input_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.sqrt((oc[0] - ic[0])**2 + (oc[1] - ic[1])**2)
        
        # Match using greedy approach
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.history[object_id].append(input_centroids[col])
            self.disappeared[object_id] = 0
            
            used_rows.add(row)
            used_cols.add(col)
        
        # Handle unmatched objects
        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # Register new objects
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col])
        
        return self._get_results()
    
    def _get_results(self) -> Dict:
        """Calculate speed metrics from tracking history."""
        speeds = []
        stopped_count = 0
        moving_count = 0
        
        for object_id, history in self.history.items():
            if len(history) >= 2:
                # Calculate displacement between last two positions
                p1 = history[-2]
                p2 = history[-1]
                displacement = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                speeds.append(displacement)
                
                # Stopped if displacement < threshold (in pixels)
                if displacement < 3:  # Less than 3 pixels = stopped
                    stopped_count += 1
                else:
                    moving_count += 1
        
        # Calculate average speed (in pixels per frame)
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        # Convert to a normalized speed scale (0-100)
        # Assuming max normal speed is ~50 pixels/frame
        normalized_speed = min(avg_speed * 2, 100)
        
        self.avg_speed = normalized_speed
        self.speeds = speeds
        
        return {
            'tracked_objects': len(self.objects),
            'avg_speed': normalized_speed,
            'avg_displacement': avg_speed,
            'stopped_count': stopped_count,
            'moving_count': moving_count,
            'speeds': speeds
        }
    
    def calculate_optical_flow(self, frame: np.ndarray) -> Dict:
        """
        Calculate optical flow for motion estimation.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Dict with flow statistics
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return {'flow_magnitude': 0.0, 'flow_direction': 0.0}
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Average magnitude (indicates overall motion)
        avg_magnitude = np.mean(magnitude)
        avg_direction = np.mean(angle)
        
        self.prev_gray = gray
        
        return {
            'flow_magnitude': float(avg_magnitude),
            'flow_direction': float(avg_direction),
            'is_slow_traffic': avg_magnitude < 1.0
        }
    
    def reset(self):
        """Reset tracker state."""
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.history = {}
        self.prev_gray = None
        self.speeds = []
        self.avg_speed = 0.0


def estimate_lane_count(frame: np.ndarray, detections: List[Dict]) -> int:
    """
    Estimate number of lanes based on vehicle positions.
    
    This is a simple heuristic based on the spread of vehicle
    x-coordinates. For accurate lane detection, use a deep learning model.
    
    Args:
        frame: Current frame
        detections: Vehicle detections
        
    Returns:
        Estimated lane count (1-6)
    """
    if not detections:
        return 2  # Default
    
    # Get horizontal positions of vehicles
    x_positions = []
    for det in detections:
        bbox = det.get('bbox', [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        x_positions.append(cx)
    
    if not x_positions:
        return 2
    
    frame_width = frame.shape[1]
    
    # Use spread of vehicles to estimate lanes
    x_min, x_max = min(x_positions), max(x_positions)
    spread = x_max - x_min
    
    # Estimate lane width as ~120-150 pixels typically
    estimated_lane_width = 130
    road_width = max(spread * 1.2, frame_width * 0.6)  # Assume vehicles span 60% min
    
    estimated_lanes = max(1, min(6, int(road_width / estimated_lane_width)))
    
    return estimated_lanes


if __name__ == "__main__":
    # Quick test
    print("Testing Motion Tracker...")
    
    tracker = MotionTracker()
    
    # Simulate detections
    detections1 = [{'bbox': [100, 100, 150, 150]}, {'bbox': [200, 200, 250, 250]}]
    detections2 = [{'bbox': [105, 102, 155, 152]}, {'bbox': [210, 205, 260, 255]}]  # Moved
    detections3 = [{'bbox': [105, 102, 155, 152]}, {'bbox': [210, 205, 260, 255]}]  # Stopped
    
    result1 = tracker.update(detections1)
    print(f"Frame 1: {result1}")
    
    result2 = tracker.update(detections2)
    print(f"Frame 2: {result2}")
    
    result3 = tracker.update(detections3)
    print(f"Frame 3: {result3}")
    
    print("\nMotion Tracker test completed!")
