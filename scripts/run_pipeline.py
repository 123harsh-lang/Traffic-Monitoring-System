"""
End-to-End Pipeline Demo

This script runs the complete traffic monitoring pipeline:
1. Video processing
2. Vehicle detection
3. Feature extraction
4. Congestion classification (rule-based)
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.detection import VehicleDetector
from src.utils import setup_logging


def run_pipeline(
    video_path: Path = None,
    max_frames: int = 50,
    use_pretrained: bool = True
):
    """
    Run the complete traffic monitoring pipeline.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process
        use_pretrained: Use pretrained classifier if available
    """
    logger = setup_logging("Pipeline")
    
    print("\n" + "="*60)
    print("AI-DRIVEN TRAFFIC MONITORING PIPELINE")
    print("="*60)
    
    # Get video path
    if video_path is None:
        # Search test_videos for mp4 files, then data_congestion for avi
        video_files = list(config.TEST_VIDEO_DIR.glob("*.mp4"))
        if not video_files:
            video_files = list(config.VIDEO_DIR.glob("*.avi"))
        if not video_files:
            logger.error("No video files found in test_videos/ or data_congestion/video/")
            return
        video_path = video_files[0]
    
    video_path = Path(video_path)
    print(f"\n📹 Video: {video_path.name}")
    
    # Step 1: Initialize detector
    print("\n" + "-"*40)
    print("STEP 1: Loading Vehicle Detector")
    print("-"*40)
    
    detector = VehicleDetector()
    print("✅ YOLO detector loaded")
    
    # Step 2: Classifier info
    print("\n" + "-"*40)
    print("STEP 2: Congestion Classifier")
    print("-"*40)
    print("ℹ️ Using rule-based classification")
    
    # Step 3: Process video
    print("\n" + "-"*40)
    print("STEP 3: Processing Video")
    print("-"*40)
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error("Could not open video")
        return
    
    frame_results = []
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or frame_count >= max_frames:
            break
        
        # Skip first frame
        if config.SKIP_FIRST_FRAME and frame_count == 0:
            frame_count += 1
            continue
        
        # Detect vehicles
        result = detector.detect(frame)
        frame_results.append(result)
        
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}: {result['vehicle_count']} vehicles detected")
        
        frame_count += 1
    
    cap.release()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Processed {len(frame_results)} frames in {elapsed:.2f}s")
    
    # Step 4: Extract features
    print("\n" + "-"*40)
    print("STEP 4: Extracting Features")
    print("-"*40)
    
    vehicle_counts = [r['vehicle_count'] for r in frame_results]
    
    total_types = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    for r in frame_results:
        for vtype, count in r['vehicle_types'].items():
            total_types[vtype] += count
    
    total_vehicles = sum(total_types.values())
    
    features = {
        'mean_vehicle_count': np.mean(vehicle_counts),
        'max_vehicle_count': np.max(vehicle_counts),
        'std_vehicle_count': np.std(vehicle_counts),
        'vehicle_density': np.mean(vehicle_counts) / 10.0,
        'car_ratio': total_types['car'] / max(total_vehicles, 1),
        'motorcycle_ratio': total_types['motorcycle'] / max(total_vehicles, 1),
        'bus_ratio': total_types['bus'] / max(total_vehicles, 1),
        'truck_ratio': total_types['truck'] / max(total_vehicles, 1),
        'hour_sin': np.sin(2 * np.pi * 12 / 24),  # Assume noon
        'hour_cos': np.cos(2 * np.pi * 12 / 24)
    }
    
    print("Extracted Features:")
    print(f"  Mean Vehicle Count: {features['mean_vehicle_count']:.2f}")
    print(f"  Max Vehicle Count:  {features['max_vehicle_count']}")
    print(f"  Std Vehicle Count:  {features['std_vehicle_count']:.2f}")
    print(f"  Vehicle Density:    {features['vehicle_density']:.4f}")
    print(f"  Vehicle Types:      {total_types}")
    
    # Step 5: Classify congestion
    print("\n" + "-"*40)
    print("STEP 5: Classifying Congestion")
    print("-"*40)
    
    predicted_class = "unknown"
    
    # Rule-based classification using per-lane thresholds
    mean_count = features['mean_vehicle_count']
    vehicles_per_lane = mean_count / config.DEFAULT_LANES
    
    if vehicles_per_lane <= config.CONGESTION_THRESHOLDS_PER_LANE["light"][1]:
        predicted_class = "light"
    elif vehicles_per_lane <= config.CONGESTION_THRESHOLDS_PER_LANE["medium"][1]:
        predicted_class = "medium"
    else:
        predicted_class = "heavy"
    
    congestion_color = config.CONGESTION_COLORS.get(predicted_class, "#FFFFFF")
    print(f"  Rule-based Prediction: {predicted_class.upper()}")
    print(f"  Vehicles per lane:     {vehicles_per_lane:.1f}")
    print(f"  (Thresholds: light≤2, medium≤5, heavy>5 per lane)")
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*60)
    print(f"  Video:           {video_path.name}")
    print(f"  Frames:          {len(frame_results)}")
    print(f"  Avg Vehicles:    {features['mean_vehicle_count']:.1f}")
    print(f"  Max Vehicles:    {features['max_vehicle_count']}")
    print(f"  Prediction:      {predicted_class.upper()}")
    print(f"  Vehicle Types:   {total_types}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Traffic Monitoring Pipeline')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='Maximum frames to process')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Use rule-based classification instead of DL')
    
    args = parser.parse_args()
    
    run_pipeline(
        video_path=Path(args.video) if args.video else None,
        max_frames=args.max_frames,
        use_pretrained=not args.no_pretrained
    )


if __name__ == "__main__":
    main()
