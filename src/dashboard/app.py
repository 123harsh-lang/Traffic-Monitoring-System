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

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Smart Traffic Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Video source selection
        st.subheader("📹 Video Source")
        
        video_options = ["Test Videos", "Upload Video"]
        video_source = st.selectbox("Select source", video_options)
        
        video_path = None
        
        if video_source == "Test Videos":
            all_videos = get_video_files()
            if all_videos:
                video_names = [v.name for v in all_videos]
                selected = st.selectbox("Select video", video_names)
                for v in all_videos:
                    if v.name == selected:
                        video_path = v
                        break
            else:
                st.warning("No test videos found in test_videos/")
        
        elif video_source == "Upload Video":
            uploaded = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])
            if uploaded:
                temp_path = config.OUTPUT_DIR / "temp_upload.mp4"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded.read())
                video_path = temp_path
        
        st.markdown("---")
        
        # Processing controls
        st.subheader("🎮 Controls")
        
        process_btn = st.button("▶️ Start Processing", use_container_width=True)
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
        reset_btn = st.button("🔄 Reset", use_container_width=True)
        
        if stop_btn:
            st.session_state.processing = False
        
        if reset_btn:
            st.session_state.count_history = []
            st.session_state.alerts = []
            st.session_state.vehicle_count = 0
            st.session_state.congestion_level = 'N/A'
            st.session_state.fps = 0.0
            st.session_state.vehicle_types = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
            st.session_state.frame_idx = 0
            st.session_state.avg_speed = 0.0
            st.session_state.traffic_flow = 'N/A'
            if st.session_state.motion_tracker:
                st.session_state.motion_tracker.reset()
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        frame_skip = st.slider("Frame Skip", 1, 10, 2)
        
        # Lane configuration
        num_lanes = st.slider("Number of Lanes", 1, 8, 2, 
            help="Set the number of lanes visible in the video")
        st.session_state.num_lanes = num_lanes
        
        st.markdown("---")
        
        # Thresholds info (per lane)
        st.subheader("📊 Congestion Thresholds")
        st.markdown(f"""
        **Per Lane** (×{num_lanes} lanes):
        - 🟢 **Light**: 0-2/lane (0-{2*num_lanes} total)
        - 🟡 **Medium**: 3-5/lane ({3*num_lanes}-{5*num_lanes} total)
        - 🔴 **Heavy**: 6+/lane ({6*num_lanes}+ total)
        
        **Speed Adjustment**:
        - Stopped traffic → upgrades level
        """)
        
        st.markdown("---")
        
        # System status - simple display
        st.subheader("📊 System Status")
        detector_status = "🟢 Active" if st.session_state.detector is not None else "🔴 Inactive"
        st.markdown(f"**Vehicle Detector:** {detector_status} | **FPS:** {st.session_state.fps:.1f}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📺 Live Feed")
        video_placeholder = st.empty()
        
        if st.session_state.current_frame is not None:
            video_placeholder.image(
                cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )
        else:
            video_placeholder.info("Select a video and click 'Start Processing'")
    
    with col2:
        st.subheader("📈 Real-time Metrics")
        
        # Congestion gauge
        CongestionGauge(st.session_state.congestion_level, key="main_gauge")
        
        # Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("🚗 Vehicles", st.session_state.vehicle_count)
        with m2:
            st.metric("⚡ FPS", f"{st.session_state.fps:.1f}")
        
        # Speed and flow metrics
        st.markdown("---")
        st.subheader("🚦 Traffic Flow")
        
        speed = st.session_state.avg_speed
        flow = st.session_state.traffic_flow
        
        # Speed indicator with color
        if flow == "Stopped":
            speed_class = "speed-stopped"
        elif flow == "Slow":
            speed_class = "speed-slow"
        else:
            speed_class = "speed-normal"
        
        st.markdown(f"""
        <div class="{speed_class} speed-metric">
            <strong>Avg Speed:</strong> {speed:.0f}<br>
            <strong>Flow:</strong> {flow}
        </div>
        """, unsafe_allow_html=True)
        
        # Moving vs stopped
        m3, m4 = st.columns(2)
        with m3:
            st.metric("🚗 Moving", st.session_state.moving_count)
        with m4:
            st.metric("🛑 Stopped", st.session_state.stopped_count)
    
    st.markdown("---")
    
    # Charts section
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📊 Vehicle Count Trend")
        if st.session_state.count_history:
            VehicleCountChart(st.session_state.count_history[-30:], key="main_chart")
        else:
            st.info("Data will appear when processing starts")
    
    with chart_col2:
        st.subheader("🚙 Vehicle Types")
        VehicleTypeDistribution(st.session_state.vehicle_types, key="main_types")
    
    st.markdown("---")
    
    # Alerts section
    st.subheader("🚨 Alerts")
    AlertBox(st.session_state.alerts[-5:] if st.session_state.alerts else [])
    
    # Processing logic
    if process_btn and video_path and video_path.exists():
        st.session_state.processing = True
        st.session_state.count_history = []
        st.session_state.frame_idx = 0
        
        # Load detector
        if st.session_state.detector is None:
            with st.spinner("Loading vehicle detector..."):
                st.session_state.detector = load_detector()
        
        # Initialize motion tracker
        st.session_state.motion_tracker = MotionTracker()
        
        # Process video
        if st.session_state.detector is not None:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            start_time = time.time()
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Accumulate types
            accumulated_types = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
            
            while cap.isOpened() and st.session_state.processing:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process frame with motion tracking
                annotated, results = process_frame(
                    frame, 
                    st.session_state.detector,
                    st.session_state.motion_tracker,
                    st.session_state.num_lanes
                )
                
                # Update accumulated types
                for vtype, count in results['vehicle_types'].items():
                    accumulated_types[vtype] += count
                
                # Update session state
                st.session_state.current_frame = annotated
                st.session_state.vehicle_count = results['vehicle_count']
                st.session_state.vehicle_types = accumulated_types.copy()
                st.session_state.congestion_level = results['congestion_level']
                st.session_state.avg_speed = results['avg_speed']
                st.session_state.stopped_count = results['stopped_count']
                st.session_state.moving_count = results['moving_count']
                st.session_state.traffic_flow = results['traffic_flow']
                st.session_state.frame_idx += 1
                
                # Update history
                st.session_state.count_history.append({
                    'time': f"F{st.session_state.frame_idx}",
                    'count': results['vehicle_count']
                })
                
                # Check for alerts
                alert = generate_alert(
                    results['congestion_level'], 
                    results['vehicle_count'],
                    results['traffic_flow']
                )
                if alert:
                    st.session_state.alerts.append(alert)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                st.session_state.fps = (frame_count + 1) / max(elapsed, 0.001)
                
                # Update video display
                video_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
                
                # Update progress
                progress_bar.progress(min(frame_count / max(total_frames, 1), 1.0))
                
                frame_count += 1
            
            cap.release()
            st.session_state.processing = False
            st.success("✅ Processing complete!")
            st.rerun()


if __name__ == "__main__":
    main()
