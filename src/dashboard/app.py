"""
Smart Traffic Monitoring Dashboard

Streamlit-based dashboard for real-time traffic monitoring,
congestion classification, and alert management.
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src import config
from src.detection import VehicleDetector, VideoProcessor, MotionTracker
from src.dashboard.components import (
    MetricCard, CongestionGauge, VideoPlayer, AlertBox,
    VehicleCountChart, VehicleTypeDistribution, CongestionHistory
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .speed-metric {
        font-size: 1.5rem;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .speed-stopped { background: rgba(255,0,0,0.2); color: #ff4444; }
    .speed-slow { background: rgba(255,165,0,0.2); color: #ffa500; }
    .speed-normal { background: rgba(0,255,0,0.2); color: #00ff00; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'detector': None,
        'motion_tracker': None,
        'processing': False,
        'current_frame': None,
        'vehicle_count': 0,
        'vehicle_types': {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0},
        'congestion_level': 'N/A',
        'alerts': [],
        'count_history': [],
        'fps': 0.0,
        'frame_idx': 0,
        'num_lanes': 2,
        'avg_speed': 0.0,
        'stopped_count': 0,
        'moving_count': 0,
        'traffic_flow': 'N/A'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_detector():
    """Load vehicle detector."""
    try:
        return VehicleDetector()
    except Exception as e:
        st.error(f"Failed to load detector: {e}")
        return None


def get_video_files():
    """Get video files from test_videos directory only."""
    videos = []
    test_video_dir = config.BASE_DIR / "test_videos"
    if test_video_dir.exists():
        for v in test_video_dir.glob("*.mp4"):
            videos.append(v)
        for v in test_video_dir.glob("*.avi"):
            videos.append(v)
        for v in test_video_dir.glob("*.mov"):
            videos.append(v)
    return videos


# =============================================================================
# CONGESTION CALCULATION (WITH SPEED)
# =============================================================================

def calculate_congestion_level(vehicle_count: int, num_lanes: int = 2, avg_speed: float = 50.0) -> tuple:
    """
    Calculate congestion level based on vehicles per lane AND traffic speed.
    
    Args:
        vehicle_count: Current vehicles in frame
        num_lanes: Number of lanes being monitored
        avg_speed: Average vehicle speed (pixels/frame, 0-100 scale)
        
    Returns:
        Tuple of (congestion_level, traffic_flow)
    """
    # Calculate vehicles per lane
    vehicles_per_lane = vehicle_count / max(num_lanes, 1)
    
    # Get per-lane thresholds from config
    thresholds = getattr(config, 'CONGESTION_THRESHOLDS_PER_LANE', {
        "light": (0, 2),
        "medium": (3, 5),
        "heavy": (6, float('inf'))
    })
    
    speed_thresholds = getattr(config, 'SPEED_THRESHOLDS', {
        "stopped": 3,
        "slow": 10,
        "normal": 25
    })
    
    # Determine base congestion from vehicle count
    if vehicles_per_lane <= thresholds["light"][1]:
        base_level = "light"
    elif vehicles_per_lane <= thresholds["medium"][1]:
        base_level = "medium"
    else:
        base_level = "heavy"
    
    # Determine traffic flow from speed
    if avg_speed < speed_thresholds["stopped"]:
        traffic_flow = "Stopped"
        # Upgrade congestion if traffic is stopped
        if base_level == "light":
            base_level = "medium"
        elif base_level == "medium":
            base_level = "heavy"
    elif avg_speed < speed_thresholds["slow"]:
        traffic_flow = "Slow"
        # Upgrade congestion if traffic is slow
        if base_level == "light" and vehicle_count > 3:
            base_level = "medium"
    else:
        traffic_flow = "Normal"
    
    return base_level, traffic_flow


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_frame(frame, detector, motion_tracker, num_lanes: int = 2):
    """
    Process a single frame through the detection pipeline with motion tracking.
    """
    from src.utils import draw_bounding_boxes, add_info_overlay
    
    # Detect vehicles
    detection_result = detector.detect(frame)
    
    # Track motion and estimate speeds
    motion_result = motion_tracker.update(detection_result['detections'])
    
    # Get average speed
    avg_speed = motion_result.get('avg_speed', 0.0)
    stopped_count = motion_result.get('stopped_count', 0)
    moving_count = motion_result.get('moving_count', 0)
    
    # Calculate congestion using both count and speed
    congestion_level, traffic_flow = calculate_congestion_level(
        detection_result['vehicle_count'], 
        num_lanes,
        avg_speed
    )
    
    # Draw annotations
    annotated_frame = draw_bounding_boxes(frame, detection_result['detections'])
    annotated_frame = add_info_overlay(
        annotated_frame,
        detection_result['vehicle_count'],
        congestion_level
    )
    
    # Add speed info to frame
    cv2.putText(
        annotated_frame,
        f"Speed: {avg_speed:.0f} | Flow: {traffic_flow}",
        (10, annotated_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return annotated_frame, {
        'vehicle_count': detection_result['vehicle_count'],
        'vehicle_types': detection_result['vehicle_types'],
        'congestion_level': congestion_level,
        'detections': detection_result['detections'],
        'avg_speed': avg_speed,
        'stopped_count': stopped_count,
        'moving_count': moving_count,
        'traffic_flow': traffic_flow
    }


def generate_alert(congestion_level: str, vehicle_count: int, traffic_flow: str) -> dict:
    """Generate alert if conditions are met."""
    if congestion_level.lower() == 'heavy':
        return {
            'type': 'warning',
            'message': f'Heavy congestion! {vehicle_count} vehicles, traffic {traffic_flow.lower()}.',
            'time': datetime.now().strftime('%H:%M:%S')
        }
    elif traffic_flow == "Stopped" and vehicle_count > 3:
        return {
            'type': 'warning',
            'message': f'Traffic stopped! {vehicle_count} vehicles detected.',
            'time': datetime.now().strftime('%H:%M:%S')
        }
    return None


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    # 1. SIDEBAR CONTROLS & CONFIGURATION
    st.sidebar.markdown("### 🎛️ Control Panel")
    
    # Get available test video feeds
    video_options = get_video_files()
    if not video_options:
        st.sidebar.error("❌ No video files found in 'test_videos/' directory.")
        st.info("Please upload or add video files (.mp4, .avi, .mov) to your 'test_videos' folder.")
        return

    # User selections
    selected_video = st.sidebar.selectbox(
        "Select Traffic Feed",
        options=video_options,
        format_func=lambda x: x.name
    )
    
    # Connect input values to session state configuration
    st.session_state['num_lanes'] = st.sidebar.slider(
        "Monitored Lanes", 
        min_value=1, 
        max_value=6, 
        value=st.session_state['num_lanes']
    )

    # Playback Control Buttons
    col_play, col_stop = st.sidebar.columns(2)
    if col_play.button("▶️ Start Stream", use_container_width=True):
        st.session_state['processing'] = True
    if col_stop.button("⏹️ Stop Stream", use_container_width=True):
        st.session_state['processing'] = False

    # 2. HEADER DISPLAY
    st.markdown(f"<div class='main-header'>{config.PAGE_TITLE}</div>", unsafe_allow_html=True)
    
    # 3. DASHBOARD METRICS LAYOUT (Top Row)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        count_placeholder = st.empty()
    with metric_col2:
        speed_placeholder = st.empty()
    with metric_col3:
        flow_placeholder = st.empty()
    with metric_col4:
        congestion_placeholder = st.empty()

    # 4. MAIN CONTENT LAYOUT (Video & Charts Area)
    content_col1, content_col2 = st.columns([2, 1])
    
    with content_col1:
        st.markdown("### 📹 Live Analytics Feed")
        # The critical structural box that makes the video move without shifting UI elements
        video_placeholder = st.empty() 
        
    with content_col2:
        st.markdown("### ⚠️ System Alerts")
        alert_placeholder = st.empty()
        st.markdown("### 📊 Distribution")
        chart_placeholder = st.empty()

    # 5. INITIALIZE TRACKING MODELS
    detector = load_detector()
    if st.session_state['motion_tracker'] is None:
        st.session_state['motion_tracker'] = MotionTracker()

    # 6. LIVE VIDEO STREAMING RUNTIME LOOP
    if st.session_state['processing'] and selected_video:
        # Open video capture pipe using OpenCV
        cap = cv2.VideoCapture(str(selected_video))
        
        while cap.isOpened() and st.session_state['processing']:
            ret, frame = cap.read()
            
            # Auto-loop video clip seamlessly when it reaches the finish frame
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Process single frame through your machine learning & motion tracking pipelines
            annotated_frame, metrics = process_frame(
                frame, 
                detector, 
                st.session_state['motion_tracker'], 
                num_lanes=st.session_state['num_lanes']
            )
            
            # Correct OpenCV BGR color standard to Web standard RGB
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # --- Live Overwrite Interface Update Cycle ---
            
            # Update the moving live video feed canvas
            video_placeholder.image(rgb_frame, use_container_width=True)
            
            # Push pipeline tracking numbers into global state
            st.session_state['vehicle_count'] = metrics['vehicle_count']
            st.session_state['vehicle_types'] = metrics['vehicle_types']
            st.session_state['congestion_level'] = metrics['congestion_level']
            st.session_state['avg_speed'] = metrics['avg_speed']
            st.session_state['stopped_count'] = metrics['stopped_count']
            st.session_state['moving_count'] = metrics['moving_count']
            st.session_state['traffic_flow'] = metrics['traffic_flow']
            
            # Log history lists for tracking graph structures
            st.session_state['count_history'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'count': metrics['vehicle_count']
            })
            # Keep history buffer capped at last 30 frames to protect browser memory
            if len(st.session_state['count_history']) > 30:
                st.session_state['count_history'].pop(0)

            # Animate the high-level custom card wrappers in real-time
            count_placeholder.metric("Vehicles Detected", f"{st.session_state['vehicle_count']} units")
            speed_placeholder.metric("Average Traffic Speed", f"{st.session_state['avg_speed']:.1f} px/f")
            flow_placeholder.metric("Current Traffic Flow", st.session_state['traffic_flow'])
            congestion_placeholder.metric("Congestion Level", st.session_state['congestion_level'].upper())
            
            # Generate and push system alerts into container block
            new_alert = generate_alert(
                st.session_state['congestion_level'],
                st.session_state['vehicle_count'],
                st.session_state['traffic_flow']
            )
            if new_alert and new_alert not in st.session_state['alerts']:
                st.session_state['alerts'].insert(0, new_alert)
                if len(st.session_state['alerts']) > 5:
                    st.session_state['alerts'].pop()
                    
            # Render your specific structural component layouts
            with alert_placeholder.container():
                for alert in st.session_state['alerts']:
                    AlertBox(alert['type'], alert['message'], alert['time'])
                    
            with chart_placeholder.container():
                VehicleTypeDistribution(st.session_state['vehicle_types'])

            # Render historical path lines below the split framework blocks
            st.markdown("### 📈 Historical Traffic Trends")
            CongestionHistory(pd.DataFrame(st.session_state['count_history']))
            
            # Pacing control interval roughly mimicking a standard 30 FPS playback index
            time.sleep(0.03)
            
        cap.release()
        
    else:
        # Safe structural fallback screen state when system is paused or idle
        video_placeholder.info("⏸️ Feed Idle. Click 'Start Stream' in the sidebar options panel to begin.")


if __name__ == "__main__":
    main()

