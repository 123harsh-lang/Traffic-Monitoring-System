"""
Dashboard UI Components

Reusable Streamlit components for the traffic monitoring dashboard.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import numpy as np


def MetricCard(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """
    Display a metric card with optional delta.
    
    Args:
        title: Metric title
        value: Current value
        delta: Change from previous value
        delta_color: Color for delta ("normal", "inverse", "off")
    """
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)


def CongestionGauge(level: str, confidence: float = None, key: str = "gauge"):
    """
    Display a congestion level gauge.
    
    Args:
        level: Congestion level ("light", "medium", "heavy")
        confidence: Optional confidence score
        key: Unique key for the chart
    """
    colors = {
        "light": "#00FF00",
        "medium": "#FFA500", 
        "heavy": "#FF0000"
    }
    
    values = {"light": 33, "medium": 66, "heavy": 100}
    
    color = colors.get(level.lower(), "#CCCCCC") if level != "N/A" else "#CCCCCC"
    value = values.get(level.lower(), 0) if level != "N/A" else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': "Congestion Level", 'font': {'size': 20, 'color': 'white'}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(0,255,0,0.2)'},
                {'range': [33, 66], 'color': 'rgba(255,165,0,0.2)'},
                {'range': [66, 100], 'color': 'rgba(255,0,0,0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': f" ({level.upper()})", 'font': {'color': 'white'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)


def VideoPlayer(frame, caption: str = None):
    """
    Display a video frame.
    
    Args:
        frame: numpy array of the frame (BGR format)
        caption: Optional caption
    """
    import cv2
    
    if frame is not None:
        # Convert BGR to RGB for display
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        st.image(frame_rgb, caption=caption, use_container_width=True)
    else:
        st.info("No video frame available")


def AlertBox(alerts: List[Dict]):
    """
    Display alert messages.
    
    Args:
        alerts: List of alert dictionaries with 'type', 'message', 'time' keys
    """
    if not alerts:
        st.success("✅ No active alerts")
        return
    
    for alert in alerts:
        alert_type = alert.get('type', 'info')
        message = alert.get('message', '')
        time = alert.get('time', '')
        
        if alert_type == 'error' or alert_type == 'critical':
            st.error(f"🚨 {time} - {message}")
        elif alert_type == 'warning':
            st.warning(f"⚠️ {time} - {message}")
        else:
            st.info(f"ℹ️ {time} - {message}")


def VehicleCountChart(data: List[Dict], title: str = "Vehicle Count Over Time", key: str = "count_chart"):
    """
    Display vehicle count time series chart.
    
    Args:
        data: List of dictionaries with 'time' and 'count' keys
        title: Chart title
        key: Unique key for the chart
    """
    if not data:
        st.info("No data available for chart")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    fig = px.line(
        df,
        x='time',
        y='count',
        title=title,
        markers=True
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font={'color': 'white'},
        xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
        yaxis={'gridcolor': 'rgba(255,255,255,0.1)'}
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)


def VehicleTypeDistribution(vehicle_types: Dict[str, int], key: str = "type_dist"):
    """
    Display vehicle type distribution as a pie chart.
    
    Args:
        vehicle_types: Dictionary mapping vehicle types to counts
        key: Unique key for the chart
    """
    if not vehicle_types or sum(vehicle_types.values()) == 0:
        st.info("No vehicle data available")
        return
    
    fig = px.pie(
        names=list(vehicle_types.keys()),
        values=list(vehicle_types.values()),
        title="Vehicle Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)


def CongestionHistory(history: List[Dict], key: str = "cong_history"):
    """
    Display congestion level history.
    
    Args:
        history: List of dictionaries with 'time' and 'level' keys
        key: Unique key for the chart
    """
    if not history:
        st.info("No congestion history available")
        return
    
    import pandas as pd
    df = pd.DataFrame(history)
    
    # Map levels to numeric values for visualization
    level_map = {'light': 1, 'medium': 2, 'heavy': 3}
    df['level_num'] = df['level'].map(level_map)
    
    fig = go.Figure()
    
    # Add bar chart with colors
    colors = df['level'].map({'light': '#00FF00', 'medium': '#FFA500', 'heavy': '#FF0000'})
    
    fig.add_trace(go.Bar(
        x=df['time'],
        y=df['level_num'],
        marker_color=colors,
        text=df['level'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Congestion History",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font={'color': 'white'},
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=['Light', 'Medium', 'Heavy'],
            gridcolor='rgba(255,255,255,0.1)'
        ),
        xaxis={'gridcolor': 'rgba(255,255,255,0.1)'}
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)


def SystemStatus(status: Dict):
    """
    Display system status indicators.
    
    Args:
        status: Dictionary with system status info
    """
    cols = st.columns(4)
    
    with cols[0]:
        detector_status = "🟢 Active" if status.get('detector_active', False) else "🔴 Inactive"
        st.markdown(f"**Vehicle Detector:** {detector_status}")
    
    with cols[1]:
        classifier_status = "🟢 Loaded" if status.get('classifier_loaded', False) else "🟡 Not Loaded"
        st.markdown(f"**Classifier:** {classifier_status}")
    
    with cols[2]:
        accident_status = "🟢 Active" if status.get('accident_detection', False) else "⚪ Disabled"
        st.markdown(f"**Accident Detection:** {accident_status}")
    
    with cols[3]:
        fps = status.get('fps', 0)
        st.markdown(f"**Processing:** {fps:.1f} FPS")
