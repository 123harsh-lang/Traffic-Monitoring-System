"""
Video Processing Module

This module handles video file processing, frame extraction,
and batch processing of video directories.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, List, Dict, Optional, Callable
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src import config
from src.detection.vehicle_detector import VehicleDetector


class VideoProcessor:
    """
    Video processing utility for traffic monitoring.
    
    Handles video reading, frame extraction, and processing with detection.
    """
    
    def __init__(
        self,
        detector: VehicleDetector = None,
        skip_first_frame: bool = None,
        frame_interval: int = None
    ):
        """
        Initialize video processor.
        
        Args:
            detector: VehicleDetector instance (created if None)
            skip_first_frame: Skip corrupted first frame
            frame_interval: Process every Nth frame
        """
        self.detector = detector or VehicleDetector()
        self.skip_first_frame = skip_first_frame if skip_first_frame is not None else config.SKIP_FIRST_FRAME
        self.frame_interval = frame_interval or config.FRAME_INTERVAL
    
    def extract_frames(
        self,
        video_path: Path,
        max_frames: int = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Yields:
            Video frames as numpy arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip first frame if configured (corrupted in UCSD dataset)
            if self.skip_first_frame and frame_count == 0:
                frame_count += 1
                continue
            
            # Apply frame interval
            if frame_count % self.frame_interval == 0:
                yield frame
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
    
    def process_video(
        self,
        video_path: Path,
        max_frames: int = None,
        callback: Callable[[np.ndarray, Dict, int], None] = None,
        show_preview: bool = False
    ) -> Dict:
        """
        Process a video and detect vehicles in each frame.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            callback: Optional callback function(frame, result, frame_idx)
            show_preview: Show live preview window
            
        Returns:
            Dictionary with aggregated results
        """
        video_path = Path(video_path)
        
        all_results = []
        total_vehicles = 0
        frame_idx = 0
        
        start_time = time.time()
        
        for frame in self.extract_frames(video_path, max_frames):
            # Detect vehicles
            result = self.detector.detect(frame)
            result['frame_idx'] = frame_idx
            all_results.append(result)
            
            total_vehicles += result['vehicle_count']
            
            # Callback if provided
            if callback:
                callback(frame, result, frame_idx)
            
            # Show preview if requested
            if show_preview:
                from src.utils import draw_bounding_boxes, add_info_overlay
                
                vis_frame = draw_bounding_boxes(frame, result['detections'])
                vis_frame = add_info_overlay(vis_frame, result['vehicle_count'])
                
                cv2.imshow('Traffic Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
        
        if show_preview:
            cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        
        # Calculate aggregated statistics
        avg_count = total_vehicles / max(len(all_results), 1)
        
        return {
            'video_path': str(video_path),
            'video_name': video_path.stem,
            'total_frames': len(all_results),
            'processing_time': elapsed_time,
            'fps': len(all_results) / max(elapsed_time, 0.001),
            'total_vehicles_detected': total_vehicles,
            'average_vehicle_count': avg_count,
            'frame_results': all_results
        }
    
    def process_directory(
        self,
        video_dir: Path,
        max_videos: int = None,
        max_frames_per_video: int = None,
        file_extension: str = ".avi"
    ) -> List[Dict]:
        """
        Process all videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            max_videos: Maximum number of videos to process
            max_frames_per_video: Maximum frames per video
            file_extension: Video file extension to look for
            
        Returns:
            List of results for each video
        """
        video_dir = Path(video_dir)
        video_files = sorted(video_dir.glob(f"*{file_extension}"))
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        all_results = []
        
        for i, video_path in enumerate(video_files):
            print(f"Processing video {i+1}/{len(video_files)}: {video_path.name}")
            
            try:
                result = self.process_video(video_path, max_frames_per_video)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                continue
        
        return all_results
    
    def get_frame_at_index(self, video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from a video.
        
        Args:
            video_path: Path to video file
            frame_idx: Frame index to retrieve
            
        Returns:
            Frame as numpy array or None if not found
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None


if __name__ == "__main__":
    # Quick test
    print("Testing Video Processor...")
    
    processor = VideoProcessor()
    
    # Test with first video in data directory
    video_dir = config.VIDEO_DIR
    video_files = list(video_dir.glob("*.avi"))
    
    if video_files:
        test_video = video_files[0]
        print(f"Testing with: {test_video}")
        
        result = processor.process_video(test_video, max_frames=5)
        print(f"Processed {result['total_frames']} frames")
        print(f"Average vehicle count: {result['average_vehicle_count']:.2f}")
    else:
        print("No video files found for testing")
    
    print("Video Processor test completed!")
