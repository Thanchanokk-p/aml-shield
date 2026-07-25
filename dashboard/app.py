"""
dashboard/app.py
=================
Streamlit dashboard for AML-Shield - a user-friendly frontend
for the FastAPI fraud detection backend running on AWS EC2.

Run:
    streamlit run dashboard/app.py
"""
import requests
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://13.40.161.73:8000"

st.set_page_config(
    page_title="AML-Shield Dashboard",
    page_icon="shield",
    layout="wide",
)

st.title("AML-Shield - Fraud Detection Dashboard")
st.caption(
    "Real-time Anti-Money Laundering transaction scoring. "
    "Powered by XGBoost + SHAP explainability."
)

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health.get("model_loaded"):
        st.success(f"Connected to API - model loaded ({API_URL})")
    else:
        st.warning("API reachable but model not loaded yet.")
except requests.exceptions.RequestException:
    st.error(f"Cannot reach API at {API_URL}. Is the EC2 instance running?")
    st.stop()

st.divider()
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount_paid = st.number_input("Amount Paid", min_value=0.0, value=50000.0, step=100.0)
    amount_received = st.number_input("Amount Received", min_value=0.0, value=50000.0, step=100.0)
    payment_format = st.selectbox(
        "Payment Format",
        ["ACH", "Wire", "Cash", "Bitcoin", "Cheque", "Credit Card", "Reinvestment"],
    )
    hour = st.slider("Hour of Day", 0, 23, 14)
    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ][x],
    )

with col2:
    is_same_bank = st.checkbox("Same Bank (sender = receiver bank)", value=False)
    is_cross_currency = st.checkbox("Cross-Currency Transaction", value=False)
    sender_tx_count = st.number_input("Sender's Historical Transaction Count", min_value=0, value=450, step=1)
    sender_avg_amount = st.number_input("Sender's Average Transaction Amount", min_value=0.0, value=500.0, step=50.0)
    threshold = st.slider(
        "Fraud Decision Threshold", 0.0, 1.0, 0.5, 0.05,
        help="Lower = catch more fraud but more false alarms. Higher = fewer false alarms but may miss fraud.",
    )

if st.button("Analyze Transaction", type="primary", use_container_width=True):
    payload = {
        "amount_paid": amount_paid,
        "amount_received": amount_received,
        "payment_format": payment_format,
        "timestamp": datetime.now().isoformat(),
        "hour": hour,
        "day_of_week": day_of_week,
        "is_same_bank": int(is_same_bank),
        "is_cross_currency": int(is_cross_currency),
        "sender_tx_count": sender_tx_count,
        "sender_avg_amount": sender_avg_amount,
    }
    with st.spinner("Scoring transaction..."):
        try:
            response = requests.post(
                f"{API_URL}/predict", params={"threshold": threshold}, json=payload, timeout=15,
            )
            response.raise_for_status()
            st.session_state["last_result"] = response.json()
            st.session_state["last_threshold"] = threshold
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    threshold = st.session_state["last_threshold"]

    st.divider()
    st.subheader("Result")

    risk_score = result["risk_score"]
    risk_level = result["risk_level"]
    flagged = result["flagged"]

    level_colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "darkred"}

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score", f"{risk_score:.4f}")
    c2.metric("Risk Level", risk_level)
    c3.metric("Flagged as Fraud", "Yes" if flagged else "No")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": level_colors.get(risk_level, "gray")},
            "steps": [
                {"range": [0, 0.3], "color": "#d4f4dd"},
                {"range": [0.3, 0.5], "color": "#ffe8b3"},
                {"range": [0.5, 0.8], "color": "#ffc9b3"},
                {"range": [0.8, 1.0], "color": "#ffb3b3"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": threshold},
        },
    ))
    st.plotly_chart(fig, use_container_width=True)

    all_reasons = sorted(result["top_reasons"], key=lambda r: abs(r["shap_value"]), reverse=True)
    show_all = st.checkbox(f"View more reasons (showing 4 of {len(all_reasons)} by default)")
    reasons = all_reasons if show_all else all_reasons[:4]

    st.subheader("Why This Decision")
    bullet_lines = []
    for r in reasons:
        direction = "increases" if r["shap_value"] > 0 else "decreases"
        bullet_lines.append(f"- **{r['label']}** - {direction} the fraud risk score")
    st.info("\n".join(bullet_lines))

    labels = [r["label"] for r in reasons]
    shap_values = [r["shap_value"] for r in reasons]
    colors = ["#C62828" if v > 0 else "#1565C0" for v in shap_values]

    fig2 = go.Figure(go.Bar(x=shap_values, y=labels, orientation="h", marker_color=colors))
    fig2.update_layout(
        title="Top Factors Driving This Decision (SHAP values)",
        xaxis_title="Impact on fraud score (red = increases, blue = decreases)",
        height=300 if not show_all else 500,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw API Response"):
        st.json(result)

st.divider()
st.caption(f"API: {API_URL} - AML-Shield v1.0")
