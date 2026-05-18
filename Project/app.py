import os
import pandas as pd
import streamlit as st

from src.predict import ThreatPredictor
from src.explain import generate_simple_explanation
from src.utils import log_prediction

st.set_page_config(
    page_title="Insider Threat Detection",
    layout="wide",
    page_icon="🛡️"
)

MODEL_PATH = "models/trained_model.pkl"
LOG_PATH = "logs/predictions.csv"

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0b0f14;
    color: #e6edf3;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}

header[data-testid="stHeader"] * {
    color: #d6dee6 !important;
}

div[data-testid="stDecoration"] {
    display: none;
}

footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #080c10;
    border-right: 1px solid #1f2933;
}

[data-testid="stSidebar"] * {
    color: #d6dee6 !important;
}

h1, h2, h3 {
    color: #f2f6fa !important;
}

.app-header {
    padding: 22px 26px;
    border: 1px solid #1f2933;
    background: linear-gradient(135deg, #111827, #0b0f14);
    border-radius: 14px;
    margin-bottom: 22px;
}

.header-kicker {
    font-family: 'JetBrains Mono', monospace;
    color: #8aa4bf;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.header-title {
    font-size: 34px;
    font-weight: 700;
    margin-top: 6px;
    margin-bottom: 8px;
}

.header-sub {
    color: #9fb1c1;
    font-size: 15px;
}

.card {
    background: #111821;
    border: 1px solid #1f2933;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 18px;
}

.card-title {
    font-family: 'JetBrains Mono', monospace;
    color: #8aa4bf;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
}

.verdict-card {
    border-radius: 16px;
    padding: 24px;
    border: 1px solid;
    background: #111821;
}

.verdict-card.normal {
    border-color: #2ea043;
    box-shadow: 0 0 0 1px rgba(46,160,67,0.15);
}

.verdict-card.malicious {
    border-color: #f85149;
    box-shadow: 0 0 0 1px rgba(248,81,73,0.15);
}

.verdict-label {
    font-family: 'JetBrains Mono', monospace;
    color: #8aa4bf;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.verdict-main {
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.verdict-main.normal {
    color: #3fb950;
}

.verdict-main.malicious {
    color: #ff7b72;
}

.confidence-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 40px;
    font-weight: 600;
}

.confidence-value.normal {
    color: #3fb950;
}

.confidence-value.malicious {
    color: #ff7b72;
}

.metric-tile {
    background: #0d131b;
    border: 1px solid #1f2933;
    border-radius: 14px;
    padding: 16px;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 25px;
    font-weight: 600;
    color: #58a6ff;
}

.metric-label {
    color: #8aa4bf;
    font-size: 13px;
    margin-top: 4px;
}

.explanation-box {
    background: #0d131b;
    border-left: 4px solid #58a6ff;
    border-radius: 10px;
    padding: 16px;
    color: #d6dee6;
    line-height: 1.7;
}

.sidebar-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #8aa4bf;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
}

.sidebar-box {
    background: #0d131b;
    border: 1px solid #1f2933;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
}

.stButton > button {
    background: #238636 !important;
    color: white !important;
    border: 1px solid #2ea043 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 0.65rem 1rem;
}

.stButton > button:hover {
    background: #2ea043 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1f2933;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header 
st.markdown("""
<div class="app-header">
    <div class="header-kicker">AI-Assisted Security Analytics</div>
    <div class="header-title">Insider Threat Detection System</div>
    <div class="header-sub">
        Behavioural record classification with confidence scoring, explanation, and audit logging.
    </div>
</div>
""", unsafe_allow_html=True)

# Model check
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run train.py first.")
    st.stop()

predictor = ThreatPredictor(MODEL_PATH)

# Session state
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

# Sidebar menu
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Navigation</div>',
        unsafe_allow_html=True
    )

    menu = st.radio(
        "Menu",
        [
            "Run Analysis",
            "Prediction Log",
            "Model Info"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-title">Behaviour Upload</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-box">
        Upload behavioural activity records for analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    input_df = None

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.success(
            f"{len(input_df)} records loaded successfully."
        )

    st.markdown("---")

    run_analysis = st.button(
        "Run Analysis"
    )

    if run_analysis and input_df is not None:

        predictions, probabilities = predictor.predict(input_df)

        result_df = input_df.copy()

        result_df["prediction"] = predictions

        result_df["prediction_label"] = (
            result_df["prediction"]
            .map({
                0: "Normal",
                1: "Malicious Insider"
            })
        )

        result_df["confidence"] = probabilities.max(axis=1)

        st.session_state.analysis_done = True
        st.session_state.input_df = input_df
        st.session_state.result_df = result_df
        st.session_state.probabilities = probabilities

    elif run_analysis and input_df is None:
        st.warning("Please upload a CSV file first.")

# Menu: Model Info 
if menu == "Model Info":
    st.markdown('<div class="card"><div class="card-title">Model Information</div>', unsafe_allow_html=True)
    st.write("Final selected model: Random Forest Classifier")
    st.write("Reason: best balance between precision, recall, and F1-score in model comparison.")
    st.write("The model is trained on behavioural employee activity features and outputs a binary classification.")
    st.markdown('</div>', unsafe_allow_html=True)

    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        st.markdown('<div class="card"><div class="card-title">Top Feature Importance</div>', unsafe_allow_html=True)
        st.dataframe(importance_df.head(10), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Menu: Prediction Log 
elif menu == "Prediction Log":
    st.markdown('<div class="card"><div class="card-title">Prediction Log</div>', unsafe_allow_html=True)

    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        st.dataframe(log_df.tail(30), use_container_width=True)
    else:
        st.info("No prediction log found yet.")

    st.markdown('</div>', unsafe_allow_html=True)

# Menu: Run Analysis
else:
    if st.session_state.input_df is None:
        st.markdown("""
        <div class="card">
            <div class="card-title">Awaiting Input</div>
            Select an input source from the sidebar, then click <b>Run Analysis</b>.
        </div>
        """, unsafe_allow_html=True)

    elif not st.session_state.analysis_done:
        st.markdown('<div class="card"><div class="card-title">Input Data Loaded</div>', unsafe_allow_html=True)
        st.write("Input data has been loaded. Click **Run Analysis** in the sidebar to classify the record.")
        st.dataframe(input_df.head(50), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        input_df = st.session_state.input_df
        result_df = st.session_state.result_df
        probabilities = st.session_state.probabilities
        total_records = len(result_df)
        malicious_count = int((result_df["prediction"] == 1).sum())
        normal_count = int((result_df["prediction"] == 0).sum())
        avg_confidence = float(result_df["confidence"].mean())

        tab1, tab2, tab3 = st.tabs([
            "Batch Summary",
            "Selected Record Analysis",
            "All Predictions"
        ])

        # Batch Summary
        with tab1:
            overall_malicious = malicious_count > 0
            verdict_class = "malicious" if overall_malicious else "normal"
            label = (
                f"{malicious_count} Potential Threat(s) Detected"
                if overall_malicious
                else "No Threats Detected"
            )

            left, right = st.columns([2, 1])

            with left:
                st.markdown(f"""
                <div class="verdict-card {verdict_class}">
                    <div class="verdict-label">Batch Classification Summary</div>
                    <div class="verdict-main {verdict_class}">{label}</div>
                </div>
                """, unsafe_allow_html=True)

            with right:
                st.markdown(f"""
                <div class="verdict-card {verdict_class}">
                    <div class="verdict-label">Average Confidence</div>
                    <div class="confidence-value {verdict_class}">{avg_confidence * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value">{total_records}</div>
                    <div class="metric-label">Total Records</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value">{normal_count}</div>
                    <div class="metric-label">Normal Records</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value">{malicious_count}</div>
                    <div class="metric-label">Threat Records</div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                threat_rate = (malicious_count / total_records) * 100 if total_records > 0 else 0
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-value">{threat_rate:.1f}%</div>
                    <div class="metric-label">Threat Rate</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">Batch Results Preview</div>', unsafe_allow_html=True)
            st.dataframe(
                result_df[["prediction_label", "confidence"] + list(input_df.columns)].head(50),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Selected Record Analysis
        with tab2:
            row_options = [
                f"Row {i} — {result_df.iloc[i]['prediction_label']} — {result_df.iloc[i]['confidence'] * 100:.2f}%"
                for i in range(len(result_df))
            ]

            selected_row_label = st.selectbox(
                "Select a record to inspect",
                row_options
            )

            selected_index = row_options.index(selected_row_label)

            selected_row = input_df.iloc[selected_index].to_dict()
            selected_prediction = int(result_df.iloc[selected_index]["prediction"])
            selected_confidence = float(result_df.iloc[selected_index]["confidence"])
            explanation = generate_simple_explanation(selected_row)

            is_malicious = selected_prediction == 1
            verdict_class = "malicious" if is_malicious else "normal"
            label = "Malicious Insider Activity" if is_malicious else "Normal Behaviour"

            left, right = st.columns([2, 1])

            with left:
                st.markdown(f"""
                <div class="verdict-card {verdict_class}">
                    <div class="verdict-label">Selected Record Classification</div>
                    <div class="verdict-main {verdict_class}">{label}</div>
                </div>
                """, unsafe_allow_html=True)

            with right:
                st.markdown(f"""
                <div class="verdict-card {verdict_class}">
                    <div class="verdict-label">Confidence Score</div>
                    <div class="confidence-value {verdict_class}">{selected_confidence * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            pages = int(selected_row.get("total_printed_pages", 0))
            off_hours = int(selected_row.get("num_printed_pages_off_hours", 0))
            burned = int(selected_row.get("total_files_burned", 0))
            entries = int(selected_row.get("num_entries", 0))
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f'<div class="metric-tile"><div class="metric-value">{pages}</div><div class="metric-label">Pages Printed</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-tile"><div class="metric-value">{off_hours}</div><div class="metric-label">Off-Hours Prints</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-tile"><div class="metric-value">{burned}</div><div class="metric-label">Files Burned</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-tile"><div class="metric-value">{entries}</div><div class="metric-label">Access Entries</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">Analysis Explanation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">Selected Record Data</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([selected_row]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            prediction_label = "Malicious" if selected_prediction == 1 else "Normal"

            if st.button("Save Selected Record to Prediction Log", use_container_width=True):
                log_prediction(
                    record=selected_row,
                    prediction=prediction_label,
                    confidence=selected_confidence,
                    explanation=explanation,
                    log_path=LOG_PATH
                )
                st.success("Selected record saved to prediction log.")

        # All Predictions 
        with tab3:
            st.markdown('<div class="card"><div class="card-title">All Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            importance_df = predictor.get_feature_importance()
            if importance_df is not None:
                st.markdown('<div class="card"><div class="card-title">Top Feature Importance</div>', unsafe_allow_html=True)
                st.dataframe(importance_df.head(10), use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)