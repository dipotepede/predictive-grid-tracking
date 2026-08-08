# predictive-grid-tracking

An interactive, edge-governed predictive grid-load tracking prototype demonstrating Statistical Process Control (SPC) governance and LiteRT execution. Designed to monitor energy consumption volatility, detect load anomalies, and enforce process control bounds across distributed power grid endpoints.

🚀 **Live Interactive Prototype:** [https://predictive-grid-tracking.streamlit.app/](https://predictive-grid-tracking.streamlit.app/)

---

## Key Features

* **Edge-Governed SPC Control Bounds:** Real-time process control monitoring utilizing Shewhart $I\text{-}mR$ charts to isolate non-random load spikes and grid instability.
* **LiteRT Edge Optimization:** Lightweight model deployment optimized via LiteRT for low-latency, real-time inference on edge grid devices.
* **Predictive Load Telemetry:** Live visualization of power demand trajectories, statistical upper/lower control limits ($\text{UCL}/\text{LCL}$), and real-time out-of-control alerts.
* **Anomalous Load Mitigation:** Automated quality gates designed to block corrupted telemetry data or uncharacteristic load surges before writing to grid management engines.
* **Interactive Telemetry Dashboard:** Streamlit interface allowing users to simulate demand surges, inspect control bounds, and evaluate load forecasting stability.

---

## Tech Stack & Architecture

* **Core Engine & Deployment:** Python, LiteRT / TensorFlow Lite, PyTorch.
* **Quality & SPC Analytics:** NumPy, pandas, Minitab-aligned $I\text{-}mR$ Shewhart Control Logic & Variance Tracking.
* **Visualization & Interface:** Streamlit, Plotly / Matplotlib.
* **Hosting & CI/CD:** Streamlit Community Cloud (Auto-deploy).

---

## Getting Started (Local Development)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/predictive-grid-tracking.git](https://github.com/your-username/predictive-grid-tracking.git)
   cd predictive-grid-tracking
