"""
Vehicle Detection Demo Script

This script demonstrates the vehicle detection capabilities
on sample videos from the UCSD Traffic Database.
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.detection import VehicleDetector, VideoProcessor
from src.utils import draw_bounding_boxes, add_info_overlay, setup_logging


def run_demo(
    video_path: Path = None,
    max_frames: int = 100,
    show_video: bool = True,
    save_output: bool = False
):
    """
    Run vehicle detection demo.
    
    Args:
        video_path: Path to video file (uses first sample if None)
        max_frames: Maximum frames to process
        show_video: Display video in window
        save_output: Save annotated video
    """
    logger = setup_logging("DetectionDemo")
    
    # Get video path
    if video_path is None:
        video_files = list(config.VIDEO_DIR.glob("*.avi"))
        if not video_files:
            logger.error("No video files found in data directory")
            return
        video_path = video_files[0]
    
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return
    
    logger.info(f"Processing video: {video_path.name}")
    
    # Initialize detector
    logger.info("Loading YOLO detector...")
    detector = VehicleDetector()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error("Could not open video")
        return
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output video writer
    writer = None
    if save_output:
        output_path = config.OUTPUT_DIR / f"demo_output_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info(f"Saving output to: {output_path}")
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    total_vehicles = 0
    
    print("\n" + "="*50)
    print("VEHICLE DETECTION DEMO")
    print("="*50)
    print(f"Video: {video_path.name}")
    print("Press 'Q' to quit")
    print("="*50 + "\n")
    
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
        
        # Draw annotations
        annotated = draw_bounding_boxes(frame, result['detections'])
        
        # Calculate FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / max(elapsed, 0.001)
        
        # Add info overlay
        annotated = add_info_overlay(
            annotated,
            result['vehicle_count'],
            "Detecting...",
            current_fps
        )
        
        # Update statistics
        total_vehicles += result['vehicle_count']
        
        # Print frame info
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: {result['vehicle_count']} vehicles | "
                  f"Types: {result['vehicle_types']} | FPS: {current_fps:.1f}")
        
        # Display
        if show_video:
            cv2.imshow('Traffic Detection Demo', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
        
        # Save output
        if writer:
            writer.write(annotated)
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # Summary
    elapsed = time.time() - start_time
    avg_vehicles = total_vehicles / max(frame_count, 1)
    
    print("\n" + "="*50)
    print("DEMO COMPLETE")
    print("="*50)
    print(f"Frames processed: {frame_count}")
    print(f"Total processing time: {elapsed:.2f}s")
    print(f"Average FPS: {frame_count/max(elapsed, 0.001):.1f}")
    print(f"Average vehicles per frame: {avg_vehicles:.1f}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection Demo')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Maximum frames to process')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable video display')
    parser.add_argument('--save', action='store_true',
                        help='Save annotated output video')
    
    args = parser.parse_args()
    
    run_demo(
        video_path=Path(args.video) if args.video else None,
        max_frames=args.max_frames,
        show_video=not args.no_display,
        save_output=args.save
    )


if __name__ == "__main__":
    main()
