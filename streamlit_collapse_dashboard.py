# streamlit_app.py
"""
Universal Collapse Control System - Interactive Dashboard
========================================================
Real-time visualization and control interface for collapse detection
and intervention across multiple domains.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from datetime import datetime, timedelta
import time
from pathlib import Path
import json

# Import our config-driven system
from config_driven_collapse_system import (
    ConfigurableCollapseSystem, 
    DEFAULT_CONFIG,
    CollapseSignature
)

# Page configuration
st.set_page_config(
    page_title="Universal Collapse Control System",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    .intervention-active {
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
    st.session_state.streaming = False
    st.session_state.data_buffer = []
    st.session_state.time_buffer = []
    st.session_state.signatures = []
    st.session_state.interventions = []
    st.session_state.config = yaml.safe_load(DEFAULT_CONFIG)

def load_sample_data(data_type):
    """Generate sample collapse data for demonstration"""
    t = np.linspace(0, 100, 1000)
    
    if data_type == "Financial Flash Crash":
        # Simulate market crash with recovery
        baseline = 100 * np.ones_like(t)
        crash = 30 / (1 + np.exp(-0.5 * (t - 40))) 
        recovery = 20 / (1 + np.exp(-0.3 * (t - 60)))
        noise = 2 * np.random.randn(len(t))
        signal = baseline - crash + recovery + noise
        
    elif data_type == "Plasma Disruption":
        # Simulate tokamak disruption
        phase1 = 0.3 / (1 + np.exp(-0.5 * (t - 30)))
        phase2 = 0.4 / (1 + np.exp(-1.0 * (t - 50)))
        oscillation = 0.2 * np.exp(-0.1 * (t - 40)) * np.cos(0.5 * t + 0.01 * t**2)
        noise = 0.05 * np.random.randn(len(t))
        signal = phase1 + phase2 + oscillation + noise
        
    elif data_type == "Neural Seizure":
        # Simulate EEG seizure pattern
        baseline = np.sin(0.2 * t) + 0.5 * np.sin(0.5 * t)
        seizure = 5 * np.exp(-((t - 50) / 10)**2) * np.sin(5 * t)
        noise = 0.3 * np.random.randn(len(t))
        signal = baseline + seizure + noise
        
    else:  # Cosmic Filament
        # Simulate density evolution
        growth = 0.1 * t
        collapse = 10 / (1 + np.exp(-0.2 * (t - 70)))
        oscillation = 2 * np.sin(0.1 * t) * np.exp(-0.01 * t)
        signal = growth + collapse + oscillation
    
    return t, signal

def create_collapse_visualization(signature: CollapseSignature, data, time_axis):
    """Create comprehensive visualization of collapse analysis"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Signal & Fit', 'Phase Space', 'Acceleration Profile',
                       'Feature Evolution', 'Intervention Window', 'Signature Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Normalize data for visualization
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # 1. Signal & Fit
    fig.add_trace(
        go.Scatter(x=time_axis, y=data_norm, name='Data', 
                  mode='markers', marker=dict(size=3, opacity=0.5)),
        row=1, col=1
    )
    
    # Add critical times
    for i, tc in enumerate(signature.critical_times):
        fig.add_vline(x=tc, line_dash="dash", line_color="red", 
                     annotation_text=f"Critical {i+1}", row=1, col=1)
    
    # 2. Phase Space
    if len(data) > 1:
        phase_x = data_norm[:-1]
        phase_y = np.gradient(data_norm)[:-1]
        
        fig.add_trace(
            go.Scatter(x=phase_x, y=phase_y, mode='markers',
                      marker=dict(size=3, color=time_axis[:-1], colorscale='Viridis'),
                      name='Phase Space'),
            row=1, col=2
        )
    
    # 3. Acceleration Profile
    accel = np.abs(np.gradient(np.gradient(data_norm)))
    fig.add_trace(
        go.Scatter(x=time_axis, y=accel, name='|d²F/dt²|', line=dict(color='red')),
        row=1, col=3
    )
    
    # 4. Feature Evolution
    # Extract key features over sliding windows
    window_size = max(50, len(data) // 20)
    stride = max(10, window_size // 5)
    
    feature_times = []
    entropies = []
    kurtoses = []
    
    for i in range(0, len(data) - window_size, stride):
        window = data[i:i+window_size]
        feature_times.append(time_axis[i + window_size//2])
        
        # Shannon entropy
        hist, _ = np.histogram(window, bins=10)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        entropies.append(-np.sum(hist * np.log2(hist)))
        
        # Kurtosis
        mean = np.mean(window)
        std = np.std(window)
        kurtoses.append(np.mean(((window - mean) / std) ** 4) - 3 if std > 0 else 0)
    
    fig.add_trace(
        go.Scatter(x=feature_times, y=entropies, name='Entropy', 
                  line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=feature_times, y=kurtoses, name='Kurtosis', 
                  line=dict(color='orange'), yaxis='y2'),
        row=2, col=1
    )
    
    # 5. Intervention Window
    start_idx, end_idx = signature.intervention_window
    window_data = data_norm[start_idx:end_idx]
    window_time = time_axis[start_idx:end_idx]
    
    fig.add_trace(
        go.Scatter(x=window_time, y=window_data, name='Intervention Window',
                  fill='tozeroy', fillcolor='rgba(0,255,0,0.3)'),
        row=2, col=2
    )
    
    # Add intervention boundaries
    fig.add_vline(x=time_axis[start_idx], line_dash="solid", line_color="green",
                 annotation_text="Start", row=2, col=2)
    fig.add_vline(x=time_axis[end_idx], line_dash="solid", line_color="green",
                 annotation_text="End", row=2, col=2)
    
    # 6. Signature Summary Table
    summary_data = [
        ["Domain", signature.domain],
        ["Collapse Type", signature.collapse_type],
        ["Phase Count", str(signature.phase_count)],
        ["Confidence", f"{signature.confidence:.2%}"],
        ["Critical Times", f"{len(signature.critical_times)}"],
        ["Alpha Exponent", f"{signature.universal_exponents.get('alpha', 'N/A')}"]
    ]
    
    fig.add_trace(
        go.Table(
            cells=dict(
                values=[[row[0] for row in summary_data],
                       [row[1] for row in summary_data]],
                align='left',
                font=dict(size=12),
                height=25
            )
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"Collapse Analysis - {signature.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        title_font_size=20
    )
    
    return fig

def main():
    st.title("🌀 Universal Collapse Control System")
    st.markdown("*Real-time detection and intervention for collapse dynamics across all domains*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Domain selection
        domain = st.selectbox(
            "Select Domain",
            ["General", "Financial Markets", "Plasma Physics", "Neuroscience", "Astrophysics"]
        )
        
        # Model configuration
        with st.expander("Model Settings", expanded=True):
            enable_oscillatory = st.checkbox("Oscillatory Chirp Model", value=True)
            enable_fractional = st.checkbox("Fractional Derivative Model", value=True)
            enable_manifold = st.checkbox("Manifold Analysis (UMAP)", value=True)
            
            st.session_state.config['models']['oscillatory']['enabled'] = enable_oscillatory
            st.session_state.config['models']['fractional']['enabled'] = enable_fractional
            st.session_state.config['features']['manifold']['enabled'] = enable_manifold
        
        # Streaming configuration
        with st.expander("Streaming Settings"):
            buffer_size = st.slider("Buffer Size", 100, 2000, 1000)
            intervention_threshold = st.slider("Intervention Threshold", 0.5, 0.95, 0.75)
            
            st.session_state.config['streaming']['buffer_size'] = buffer_size
            st.session_state.config['streaming']['intervention_threshold'] = intervention_threshold
        
        # Export configuration
        if st.button("💾 Export Configuration"):
            config_yaml = yaml.dump(st.session_state.config, default_flow_style=False)
            st.download_button(
                label="Download config.yaml",
                data=config_yaml,
                file_name="collapse_config.yaml",
                mime="text/yaml"
            )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📡 Real-Time Monitor", 
                                       "🔍 Signature Library", "📚 Documentation"])
    
    with tab1:
        st.header("Signal Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Data input options
            data_source = st.radio(
                "Data Source",
                ["Upload File", "Sample Data", "Live Stream"],
                horizontal=True
            )
            
            if data_source == "Upload File":
                uploaded_file = st.file_uploader(
                    "Choose a file", 
                    type=['csv', 'txt', 'npy']
                )
                
                if uploaded_file is not None:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        data = df.iloc[:, -1].values  # Assume last column is signal
                        time_axis = df.iloc[:, 0].values if df.shape[1] > 1 else np.arange(len(data))
                    elif uploaded_file.name.endswith('.npy'):
                        data = np.load(uploaded_file)
                        time_axis = np.arange(len(data))
                    else:
                        data = np.loadtxt(uploaded_file)
                        time_axis = np.arange(len(data))
                    
                    if st.button("🔬 Analyze Signal"):
                        with st.spinner("Analyzing collapse dynamics..."):
                            # Initialize system
                            if st.session_state.system is None:
                                st.session_state.system = ConfigurableCollapseSystem()
                            
                            # Process signal
                            signature = st.session_state.system.process_signal(
                                data, time_axis, domain.lower()
                            )
                            st.session_state.signatures.append(signature)
                            
                            # Display results
                            st.success("✅ Analysis complete!")
                            
                            # Visualize
                            fig = create_collapse_visualization(signature, data, time_axis)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Collapse Type", signature.collapse_type)
                            col2.metric("Phase Count", signature.phase_count)
                            col3.metric("Confidence", f"{signature.confidence:.2%}")
                            col4.metric("Intervention Window", 
                                       f"{signature.intervention_window[0]}-{signature.intervention_window[1]}")
                            
                            # Download results
                            st.download_button(
                                label="📥 Download Signature",
                                data=json.dumps(signature.to_dict(), indent=2, default=str),
                                file_name=f"signature_{signature.timestamp:%Y%m%d_%H%M%S}.json",
                                mime="application/json"
                            )
            
            elif data_source == "Sample Data":
                sample_type = st.selectbox(
                    "Select Sample Type",
                    ["Financial Flash Crash", "Plasma Disruption", 
                     "Neural Seizure", "Cosmic Filament"]
                )
                
                if st.button("🎲 Generate & Analyze"):
                    time_axis, data = load_sample_data(sample_type)
                    
                    with st.spinner("Analyzing sample data..."):
                        if st.session_state.system is None:
                            st.session_state.system = ConfigurableCollapseSystem()
                        
                        signature = st.session_state.system.process_signal(
                            data, time_axis, sample_type.lower()
                        )
                        st.session_state.signatures.append(signature)
                        
                        fig = create_collapse_visualization(signature, data, time_axis)
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Quick Actions")
            
            if st.button("🔄 Reset System"):
                st.session_state.system = None
                st.session_state.signatures = []
                st.session_state.interventions = []
                st.success("System reset!")
            
            if len(st.session_state.signatures) > 0:
                st.subheader("📊 Recent Signatures")
                for sig in st.session_state.signatures[-3:]:
                    with st.expander(f"{sig.domain} - {sig.timestamp:%H:%M:%S}"):
                        st.write(f"Type: {sig.collapse_type}")
                        st.write(f"Phases: {sig.phase_count}")
                        st.write(f"Confidence: {sig.confidence:.2%}")
    
    with tab2:
        st.header("Real-Time Monitoring")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Real-time plot placeholder
            plot_placeholder = st.empty()
            
            # Metrics placeholders
            metric_cols = st.columns(5)
            metric_placeholders = [col.empty() for col in metric_cols]
            
            # Alert placeholder
            alert_placeholder = st.empty()
        
        with col2:
            if st.button("▶️ Start Monitoring" if not st.session_state.streaming else "⏸️ Stop Monitoring"):
                st.session_state.streaming = not st.session_state.streaming
            
            if st.session_state.streaming:
                st.info("🟢 Monitoring Active")
                
                # Simulate real-time data
                if 'stream_index' not in st.session_state:
                    st.session_state.stream_index = 0
                
                # Generate and process streaming data
                while st.session_state.streaming:
                    # Generate new data point
                    t = st.session_state.stream_index * 0.1
                    
                    # Simulate collapse dynamics
                    if t < 30:
                        value = np.sin(0.5 * t) + 0.1 * np.random.randn()
                    elif t < 50:
                        value = 2 / (1 + np.exp(-0.5 * (t - 40))) + 0.1 * np.random.randn()
                    else:
                        value = 2 + np.sin(0.3 * t) + 0.1 * np.random.randn()
                    
                    st.session_state.data_buffer.append(value)
                    st.session_state.time_buffer.append(t)
                    
                    # Maintain buffer size
                    if len(st.session_state.data_buffer) > 200:
                        st.session_state.data_buffer.pop(0)
                        st.session_state.time_buffer.pop(0)
                    
                    # Update plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=st.session_state.time_buffer,
                        y=st.session_state.data_buffer,
                        mode='lines',
                        name='Signal',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Real-Time Signal Monitor",
                        xaxis_title="Time",
                        yaxis_title="Signal Value",
                        height=400
                    )
                    
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    # Update metrics
                    if len(st.session_state.data_buffer) > 50:
                        recent_data = np.array(st.session_state.data_buffer[-50:])
                        
                        metric_placeholders[0].metric("Mean", f"{np.mean(recent_data):.3f}")
                        metric_placeholders[1].metric("Std Dev", f"{np.std(recent_data):.3f}")
                        metric_placeholders[2].metric("Kurtosis", 
                                                     f"{st.session_state.system._kurtosis(recent_data):.2f}")
                        
                        # Simple entropy calculation
                        hist, _ = np.histogram(recent_data, bins=10)
                        hist = hist / np.sum(hist)
                        hist = hist[hist > 0]
                        entropy = -np.sum(hist * np.log2(hist))
                        metric_placeholders[3].metric("Entropy", f"{entropy:.3f}")
                        
                        # Collapse probability (simplified)
                        collapse_prob = min(1.0, max(0, 
                            (st.session_state.system._kurtosis(recent_data) / 10) +
                            (1 - entropy / 3)
                        ))
                        metric_placeholders[4].metric("Collapse Risk", f"{collapse_prob:.1%}")
                        
                        # Alert if high risk
                        if collapse_prob > 0.75:
                            alert_placeholder.error(
                                f"⚠️ HIGH COLLAPSE RISK DETECTED: {collapse_prob:.1%}"
                            )
                        elif collapse_prob > 0.5:
                            alert_placeholder.warning(
                                f"⚠️ Elevated collapse risk: {collapse_prob:.1%}"
                            )
                        else:
                            alert_placeholder.empty()
                    
                    st.session_state.stream_index += 1
                    time.sleep(0.1)  # Simulate real-time delay
    
    with tab3:
        st.header("Signature Library")
        
        if len(st.session_state.signatures) > 0:
            # Create signature comparison
            sig_data = []
            for sig in st.session_state.signatures:
                sig_data.append({
                    'Timestamp': sig.timestamp,
                    'Domain': sig.domain,
                    'Type': sig.collapse_type,
                    'Phases': sig.phase_count,
                    'Confidence': sig.confidence,
                    'Alpha': sig.universal_exponents.get('alpha', 'N/A')
                })
            
            df = pd.DataFrame(sig_data)
            
            # Display statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Signature Statistics")
                st.dataframe(df)
                
                # Download all signatures
                if st.button("📥 Export All Signatures"):
                    all_sigs = [sig.to_dict() for sig in st.session_state.signatures]
                    st.download_button(
                        label="Download signatures.json",
                        data=json.dumps(all_sigs, indent=2, default=str),
                        file_name=f"all_signatures_{datetime.now():%Y%m%d_%H%M%S}.json",
                        mime="application/json"
                    )
            
            with col2:
                st.subheader("Collapse Type Distribution")
                type_counts = df['Type'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                           title="Collapse Types")
                st.plotly_chart(fig)
                
                st.subheader("Domain Distribution")
                domain_counts = df['Domain'].value_counts()
                fig = px.bar(x=domain_counts.index, y=domain_counts.values,
                           title="Signatures by Domain")
                st.plotly_chart(fig)
        else:
            st.info("No signatures collected yet. Analyze some signals to build your library!")
    
    with tab4:
        st.header("Documentation")
        
        st.markdown("""
        ## 🌀 Universal Collapse Control System
        
        This system implements a unified framework for detecting, analyzing, and intervening in 
        collapse dynamics across multiple domains:
        
        ### Key Features:
        
        - **🔬 Multi-Model Analysis**: Oscillatory chirp, fractional derivatives, and manifold learning
        - **📊 Comprehensive Features**: Entropy, wavelets, EMD/VMD, topological analysis
        - **⚡ Real-Time Monitoring**: Stream processing with intervention triggers
        - **🌐 Cross-Domain**: Financial, plasma, neural, and astrophysical applications
        
        ### Collapse Types:
        
        1. **Sharp Transition**: High kurtosis, sudden phase change
        2. **Ordered Emergence**: Low entropy, structure formation
        3. **Multi-Scale Cascade**: Multiple IMFs, hierarchical collapse
        4. **Gradual Evolution**: Smooth transition, predictable trajectory
        
        ### Universal Exponents:
        
        - **α (Alpha)**: Power-law scaling near critical points
        - **β (Beta)**: Order parameter evolution
        - **γ (Gamma)**: Correlation length divergence
        
        ### Intervention Strategies:
        
        - **Pressure Relief**: For high-kurtosis sharp transitions
        - **Entropy Injection**: For over-ordered systems
        - **Topology Restructuring**: For connectivity breakdown
        - **Adaptive Damping**: General stabilization
        
        ### Configuration:
        
        The system is fully configurable via YAML files. Key parameters include:
        - Model selection and hyperparameters
        - Feature extraction methods
        - Streaming buffer sizes
        - Intervention thresholds
        
        ### API Usage:
        
        ```python
        # Initialize system
        system = ConfigurableCollapseSystem('config.yaml')
        
        # Analyze signal
        signature = system.process_signal(data, timestamps, domain='financial')
        
        # Stream processing
        for intervention in system.stream_process(data_generator):
            print(f"Intervention: {intervention.intervention_type}")
        ```
        
        For more information, see the [GitHub repository](https://github.com/your-repo/collapse-control).
        """)

if __name__ == "__main__":
    main()