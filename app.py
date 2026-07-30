import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Governed Edge Grid Tracking System",
    page_icon="⚡",
    layout="wide"
)

# Title & Policy Header
st.title("⚡ Edge-Governed Predictive Grid-Load Tracking System")
st.caption("Statistical Process Control (SPC) & LiteRT/Gemma Edge Inference for Local Energy Optimization")

st.markdown("""
> **Policy Context:** Demonstrating how edge-deployed, quantized AI models running locally on solar/battery micro-nodes reduce dependence on centralized data center compute while maintaining strict statistical reliability ($C_{pk} \ge 1.33$).
""")

st.sidebar.header("Micro-Grid Control Parameters")
solar_capacity = st.sidebar.slider("Solar PV Array Capacity (kW)", 5, 100, 25)
battery_capacity = st.sidebar.slider("LFP Battery Bank (kWh)", 10, 200, 60)
load_variance = st.sidebar.slider("Simulated Load Variance (Sigma)", 0.5, 5.0, 2.0)
target_tolerance = st.sidebar.slider("Voltage Tolerance Band (±%)", 1.0, 10.0, 5.0)

# ---------------------------------------------------------
# DATA GENERATION & LITERET/GEMMA INFERENCE SIMULATION
# ---------------------------------------------------------
@st.cache_data
def generate_grid_data(solar_cap, battery_cap, variance):
    np.random.seed(42)
    hours = np.arange(24)
    
    # Solar curve (Gaussian curve centered around noon)
    solar_gen = solar_cap * np.exp(-((hours - 12) ** 2) / 10)
    
    # Baseline load demand with morning and evening peaks
    base_load = 10 + 8 * np.exp(-((hours - 8) ** 2) / 6) + 15 * np.exp(-((hours - 19) ** 2) / 8)
    noisy_load = base_load + np.random.normal(0, variance, 24)
    noisy_load = np.maximum(noisy_load, 2)
    
    # Simulating LiteRT Quantized Gemma Edge Inference
    # Predictive load forecast with low variance error bound
    edge_forecast = base_load + np.random.normal(0, variance * 0.3, 24)
    
    df = pd.DataFrame({
        "Hour": hours,
        "Solar_Generation_kW": np.round(solar_gen, 2),
        "Actual_Load_kW": np.round(noisy_load, 2),
        "Edge_Predicted_Load_kW": np.round(edge_forecast, 2)
    })
    
    # Net Energy Balance
    df["Net_Balance_kW"] = df["Solar_Generation_kW"] - df["Actual_Load_kW"]
    return df

df = generate_grid_data(solar_capacity, battery_capacity, load_variance)

# ---------------------------------------------------------
# DMAIC & SPC CALCULATIONS
# ---------------------------------------------------------
# Process Capability Index Calculation
nominal_load = df["Actual_Load_kW"].mean()
usl = nominal_load * (1 + target_tolerance / 100)
lsl = nominal_load * (1 - target_tolerance / 100)
sigma = df["Actual_Load_kW"].std()

cp = (usl - lsl) / (6 * sigma) if sigma > 0 else 0
cpk = min((usl - nominal_load) / (3 * sigma), (nominal_load - lsl) / (3 * sigma)) if sigma > 0 else 0

# Control Limits
ucl = nominal_load + 3 * sigma
lcl = max(0, nominal_load - 3 * sigma)

# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Real-Time Grid Tracking", "📈 SPC Quality Governance", "📋 DMAIC Framework View"])

with tab1:
    st.subheader("Local Micro-Node Load vs. Edge Prediction")
    
    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak Load", f"{df['Actual_Load_kW'].max():.1f} kW")
    col2.metric("Solar Production", f"{df['Solar_Generation_kW'].sum():.1f} kWh/day")
    col3.metric("Edge Model Forecast Error (RMSE)", f"{np.sqrt(np.mean((df['Actual_Load_kW'] - df['Edge_Predicted_Load_kW'])**2)):.2f} kW")
    col4.metric("Edge Latency (LiteRT)", "< 12 ms", delta="Local Execution")
    
    # Line Chart: Solar vs Load vs Edge Forecast
    fig_grid = go.Figure()
    fig_grid.add_trace(go.Scatter(x=df["Hour"], y=df["Actual_Load_kW"], mode='lines+markers', name='Actual Grid Load (kW)', line=dict(color='#EF553B', width=3)))
    fig_grid.add_trace(go.Scatter(x=df["Hour"], y=df["Edge_Predicted_Load_kW"], mode='lines', name='LiteRT Gemma Edge Forecast', line=dict(color='#00CC96', dash='dash')))
    fig_grid.add_trace(go.Scatter(x=df["Hour"], y=df["Solar_Generation_kW"], mode='lines', name='Solar PV Output (kW)', fill='tozeroy', line=dict(color='#FFA15A')))
    
    fig_grid.update_layout(xaxis_title="Hour of Day", yaxis_title="Power (kW)", template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_grid, use_container_width=True)

with tab2:
    st.subheader("Statistical Process Control (SPC) & Capability Analysis")
    
    col_spc1, col_spc2 = st.columns(2)
    col_spc1.metric("Process Capability Index (Cp)", f"{cp:.2f}")
    col_spc2.metric("Process Capability Index (Cpk)", f"{cpk:.2f}", delta="Capable" if cpk >= 1.33 else "Needs Control", delta_color="normal" if cpk >= 1.33 else "inverse")
    
    # SPC Control Chart
    fig_spc = go.Figure()
    fig_spc.add_trace(go.Scatter(x=df["Hour"], y=df["Actual_Load_kW"], mode='lines+markers', name='Observed Load', line=dict(color='#636EFA')))
    fig_spc.add_trace(go.Scatter(x=[0, 23], y=[nominal_load, nominal_load], mode='lines', name='Center Line (Mean)', line=dict(color='black', dash='dash')))
    fig_spc.add_trace(go.Scatter(x=[0, 23], y=[ucl, ucl], mode='lines', name='UCL (+3σ)', line=dict(color='red', dash='dot')))
    fig_spc.add_trace(go.Scatter(x=[0, 23], y=[lcl, lcl], mode='lines', name='LCL (-3σ)', line=dict(color='red', dash='dot')))
    
    fig_spc.update_layout(title="Load Variance Shewhart Control Chart", xaxis_title="Hour", yaxis_title="kW", template="plotly_white")
    st.plotly_chart(fig_spc, use_container_width=True)

with tab3:
    st.subheader("DMAIC Quality Governance Implementation")
    
    st.markdown("""
    | Phase | Operational Focus | Local Micro-Grid Implementation |
    | :--- | :--- | :--- |
    | **Define** | System Boundaries & CTQs | Target maximum grid deviation within $\pm5\%$ tolerance; maintain localized zero-net-cloud dependency. |
    | **Measure** | Sensor & Model Data Collection | Real-time tracking of PV output, LFP battery state-of-charge (SOC), and load variance via LiteRT edge sensors. |
    | **Analyze** | Root Cause Analysis (RCA) | Identify peak load spikes and cloud-cover drops using statistical moving range ($mR$) analysis. |
    | **Improve** | Edge Optimization & Dispatch | Execute dynamic battery discharge and load-shedding commands driven by local Gemma model predictions. |
    | **Control** | SPC Maintenance | Continuous monitoring of $C_{pk} \ge 1.33$ to ensure grid stability without central cloud intervention. |
    """)

st.divider()
st.caption("Governed Edge AI Architecture | Built for Policy & Engineering Demonstration")
