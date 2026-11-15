
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import wfdb

from pathlib import Path
from io import BytesIO
from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy.signal import resample, find_peaks
from matplotlib.patches import Wedge
from tempfile import TemporaryDirectory

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as PDFImage
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

import warnings
warnings.filterwarnings("ignore")


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ECG-Trust AI",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


# ==================================================
# 2. DASHBOARD DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #172235;
}
h1, h2, h3 {
    color: #60a5fa;
}
.stButton button {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("❤️ ECG-Trust AI")

st.caption(
    "Adaptive ECG Classification with Explanation "
    "Stability and Reliability Assessment"
)

st.write(
    "An experimental AI research prototype for ECG "
    "heartbeat classification and reliability assessment."
)


# ==================================================
# 3. LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    cnn = tf.keras.models.load_model(
        RESULTS_DIR / "mitbih_1d_cnn.keras"
    )

    rf = joblib.load(
        RESULTS_DIR / "rf_corrected.pkl"
    )

    xgb = joblib.load(
        RESULTS_DIR / "xgb_corrected.pkl"
    )

    encoder = joblib.load(
        RESULTS_DIR / "ecg_label_encoder.pkl"
    )

    return cnn, rf, xgb, encoder


try:

    cnn_model, rf_model, xgb_model, label_encoder = load_models()

    st.sidebar.success("All AI models loaded successfully")

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


CLASS_DESCRIPTIONS = {
    "N": "Normal beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown or unclassified beat"
}


# ==================================================
# 4. FEATURE EXTRACTION
# ==================================================

def extract_features(signal):

    signal = np.asarray(signal).flatten()

    features = [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.ptp(signal),
        np.sqrt(np.mean(signal ** 2)),
        np.mean(np.abs(signal)),
        skew(signal),
        kurtosis(signal),
        np.sum(signal ** 2),
        np.mean(np.diff(signal)),
        np.std(np.diff(signal)),
        np.max(np.abs(np.diff(signal))),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        np.mean(np.abs(np.diff(signal, n=2))),
        np.sum(np.diff(np.sign(signal)) != 0)
    ]

    return np.nan_to_num(
        np.array(features).reshape(1, -1),
        nan=0,
        posinf=0,
        neginf=0
    )


# ==================================================
# 5. SIGNAL QUALITY SCORE
# ==================================================

def calculate_sqs(signal):

    signal = np.asarray(signal).flatten()

    baseline_ratio = (
        abs(np.mean(signal)) /
        (np.std(signal) + 1e-8)
    )

    high_frequency = np.mean(np.diff(signal) ** 2)

    total_energy = np.mean(signal ** 2) + 1e-8

    high_ratio = high_frequency / total_energy

    sqs = 1 / (1 + baseline_ratio + high_ratio)

    return float(np.clip(sqs, 0, 1))


# ==================================================
# 6. INTEGRATED GRADIENTS
# ==================================================

def integrated_gradients(signal, steps=32):

    x = tf.convert_to_tensor(
        signal.reshape(1, 252, 1),
        dtype=tf.float32
    )

    baseline = tf.zeros_like(x)

    gradients = []

    for i in range(steps + 1):

        item = baseline + (
            float(i) / steps
        ) * (x - baseline)

        with tf.GradientTape() as tape:

            tape.watch(item)

            prediction = cnn_model(
                item,
                training=False
            )

            predicted_class = tf.argmax(
                prediction[0]
            )

            score = prediction[:, predicted_class]

        gradient = tape.gradient(score, item)

        gradients.append(gradient)

    average_gradients = tf.reduce_mean(
        tf.stack(gradients[:-1]),
        axis=0
    )

    attribution = (
        x - baseline
    ) * average_gradients

    return np.abs(
        attribution.numpy().flatten()
    )


# ==================================================
# 7. ESS
# ==================================================

def calculate_ess(signal):

    clean = np.asarray(signal).flatten()

    noise_power = (
        np.mean(clean ** 2) / 10
    )

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        clean.shape
    )

    noisy = clean + noise

    clean_attr = integrated_gradients(clean)
    noisy_attr = integrated_gradients(noisy)

    clean_mask = (
        clean_attr >= np.percentile(clean_attr, 80)
    )

    noisy_mask = (
        noisy_attr >= np.percentile(noisy_attr, 80)
    )

    intersection = np.logical_and(
        clean_mask,
        noisy_mask
    ).sum()

    union = np.logical_or(
        clean_mask,
        noisy_mask
    ).sum()

    iou = intersection / (union + 1e-8)

    if (
        np.std(clean_attr) == 0
        or np.std(noisy_attr) == 0
    ):
        correlation = 0
    else:
        correlation = np.corrcoef(
            clean_attr,
            noisy_attr
        )[0, 1]

    if not np.isfinite(correlation):
        correlation = 0

    correlation_score = (correlation + 1) / 2

    ess = 0.5 * iou + 0.5 * correlation_score

    return float(np.clip(ess, 0, 1)), clean_attr


# ==================================================
# 8. HEARTBEAT ANALYSIS
# ==================================================

def analyse_heartbeat(signal):

    signal = np.asarray(signal).flatten()

    if len(signal) != 252:
        raise ValueError("Heartbeat must contain 252 samples")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Invalid signal values")

    probabilities = cnn_model.predict(
        signal.reshape(1, 252, 1),
        verbose=0
    )[0]

    class_index = int(np.argmax(probabilities))

    prediction = str(
        label_encoder.inverse_transform(
            [class_index]
        )[0]
    )

    confidence = float(probabilities[class_index])

    sqs = calculate_sqs(signal)

    ess, attribution = calculate_ess(signal)

    ers = (
        0.30 * sqs +
        0.30 * confidence +
        0.40 * ess
    )

    rf_prediction = "Unavailable"
    xgb_prediction = "Unavailable"

    try:

        features = extract_features(signal)

        if (
            hasattr(rf_model, "n_features_in_")
            and features.shape[1] == rf_model.n_features_in_
        ):
            rf_prediction = str(
                rf_model.predict(features)[0]
            )

        if (
            hasattr(xgb_model, "n_features_in_")
            and features.shape[1] == xgb_model.n_features_in_
        ):
            xgb_prediction = str(
                xgb_model.predict(features)[0]
            )

    except Exception:
        pass

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "sqs": sqs,
        "ess": ess,
        "ers": ers,
        "attribution": attribution,
        "rf": rf_prediction,
        "xgb": xgb_prediction
    }


# ==================================================
# 9. ECG PLOTS
# ==================================================

def create_waveform_plot(signal):

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(signal, linewidth=1.5)

    ax.set_title("ECG Heartbeat Waveform")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig


def create_explanation_plot(signal, attribution):

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(
        signal,
        label="ECG Signal"
    )

    threshold = np.percentile(
        attribution,
        80
    )

    ax.fill_between(
        np.arange(len(signal)),
        signal,
        where=attribution >= threshold,
        alpha=0.35,
        label="Important regions"
    )

    ax.set_title("Integrated Gradients Explanation")
    ax.set_xlabel("Sample")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig


# ==================================================
# 10. ERS GAUGE
# ==================================================

def create_ers_gauge(score):

    score = float(np.clip(score, 0, 1))

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.set_aspect("equal")

    ax.add_patch(
        Wedge(
            (0, 0),
            1,
            0,
            180,
            width=0.22,
            facecolor="#334155"
        )
    )

    ax.add_patch(
        Wedge(
            (0, 0),
            1,
            0,
            180 * score,
            width=0.22,
            facecolor="#22c55e"
        )
    )

    ax.text(
        0,
        0.12,
        f"{score * 100:.1f}%",
        ha="center",
        fontsize=25,
        fontweight="bold",
        color="#111827"
    )

    ax.text(
        0,
        -0.12,
        "ECG Reliability Score",
        ha="center",
        fontsize=12,
        color="#334155"
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.25, 1.2)
    ax.axis("off")

    fig.tight_layout()

    return fig


# ==================================================
# 11. PDF REPORT
# ==================================================

def generate_pdf_report(result, waveform_fig):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = [
        Paragraph(
            "ECG-Trust AI - Analysis Report",
            styles["Title"]
        ),
        Spacer(1, 15)
    ]

    table_data = [
        ["Parameter", "Result"],
        ["Predicted Class", result["prediction"]],
        ["CNN Confidence", f"{result['confidence'] * 100:.2f}%"],
        ["SQS", f"{result['sqs']:.4f}"],
        ["ESS", f"{result['ess']:.4f}"],
        ["ERS", f"{result['ers'] * 100:.2f}%"],
        ["Random Forest", result["rf"]],
        ["XGBoost", result["xgb"]]
    ]

    table = Table(table_data, colWidths=[220, 220])

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 20))

    image_buffer = BytesIO()

    waveform_fig.savefig(
        image_buffer,
        format="png",
        bbox_inches="tight"
    )

    image_buffer.seek(0)

    elements.append(
        PDFImage(
            image_buffer,
            width=450,
            height=140
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "Experimental research prototype. "
            "Not clinically validated and not intended "
            "for medical diagnosis.",
            styles["Normal"]
        )
    )

    document.build(elements)

    buffer.seek(0)

    return buffer


# ==================================================
# 12. CSV HEARTBEAT LOADER
# ==================================================

def read_heartbeat_csv(uploaded_file):

    df = pd.read_csv(
        uploaded_file,
        header=None
    )

    values = pd.to_numeric(
        df.values.flatten(),
        errors="coerce"
    )

    values = values[np.isfinite(values)]

    if len(values) != 252:
        raise ValueError(
            f"Expected 252 samples but found {len(values)}"
        )

    return values.astype(float)


# ==================================================
# 13. RAW ECG HEARTBEAT EXTRACTION
# ==================================================

AAMI_SYMBOLS = {
    "N", "L", "R", "e", "j",
    "A", "a", "J", "S",
    "V", "E", "F",
    "/", "f", "Q"
}


def extract_heartbeats(ecg, fs, positions):

    ecg = np.asarray(ecg).flatten()

    before = int(0.25 * fs)
    after = int(0.45 * fs)

    beats = []
    valid_positions = []

    for position in positions:

        start = int(position) - before
        end = int(position) + after

        if start < 0 or end > len(ecg):
            continue

        beat = ecg[start:end]

        if len(beat) != 252:
            beat = resample(beat, 252)

        if not np.all(np.isfinite(beat)):
            continue

        beats.append(
            beat.astype(np.float32)
        )

        valid_positions.append(int(position))

    if not beats:

        return (
            np.empty((0, 252), dtype=np.float32),
            []
        )

    return np.asarray(beats), valid_positions


# ==================================================
# 14. AUTOMATIC R-PEAK DETECTION
# ==================================================

def detect_r_peaks(ecg, fs):

    ecg = np.asarray(ecg, dtype=float).flatten()

    if fs <= 0:
        raise ValueError("Invalid sampling rate")

    if len(ecg) < int(fs * 2):
        raise ValueError("Recording is too short")

    if not np.all(np.isfinite(ecg)):
        raise ValueError("Recording contains invalid values")

    centered = ecg - np.median(ecg)

    peaks, _ = find_peaks(
        np.abs(centered),
        distance=max(1, int(0.25 * fs)),
        prominence=max(
            0.5 * np.std(centered),
            1e-8
        )
    )

    return peaks


# ==================================================
# 15. FULL CSV RECORDING LOADER
# ==================================================

def load_csv_recording(uploaded_file, column, fs):

    df = pd.read_csv(uploaded_file)

    if column not in df.columns:
        raise ValueError("ECG column not found")

    ecg = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    positions = detect_r_peaks(ecg, fs)

    beats, valid_positions = extract_heartbeats(
        ecg,
        fs,
        positions
    )

    return beats, valid_positions, fs, "Automatic detection"


# ==================================================
# 16. FULL WFDB RECORDING LOADER
# ==================================================

def load_wfdb_recording(uploaded_files):

    file_map = {
        Path(f.name).suffix.lower(): f
        for f in uploaded_files
    }

    if ".hea" not in file_map or ".dat" not in file_map:
        raise ValueError("Upload matching .hea and .dat files")

    with TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        for uploaded_file in uploaded_files:

            output_path = (
                temp_path / Path(uploaded_file.name).name
            )

            output_path.write_bytes(
                uploaded_file.getvalue()
            )

        record_name = Path(
            file_map[".hea"].name
        ).stem

        record_path = str(
            temp_path / record_name
        )

        record = wfdb.rdrecord(record_path)

        ecg = record.p_signal[:, 0]

        fs = record.fs

        if ".atr" in file_map:

            annotation = wfdb.rdann(
                record_path,
                "atr"
            )

            positions = [
                pos
                for pos, symbol in zip(
                    annotation.sample,
                    annotation.symbol
                )
                if symbol in AAMI_SYMBOLS
            ]

            method = "WFDB atr annotations"

        else:

            positions = detect_r_peaks(ecg, fs)

            method = "Experimental automatic detection"

        beats, valid_positions = extract_heartbeats(
            ecg,
            fs,
            positions
        )

        return (
            beats,
            valid_positions,
            fs,
            method
        )


# ==================================================
# 17. RECORDING CLASSIFICATION
# ==================================================

def analyse_recording(beats, positions):

    if len(beats) == 0:
        raise ValueError("No valid heartbeat segments found")

    # CNN batch prediction

    probabilities = cnn_model.predict(
        beats.reshape(-1, 252, 1),
        verbose=0,
        batch_size=64
    )

    rows = []

    progress = st.progress(0)

    for index, signal in enumerate(beats):

        probs = probabilities[index]

        class_index = int(np.argmax(probs))

        prediction = str(
            label_encoder.inverse_transform(
                [class_index]
            )[0]
        )

        confidence = float(probs[class_index])

        sqs = calculate_sqs(signal)

        # Explanation stability is computationally expensive.
        # Calculate it for the first 20 beats in this prototype.
        if index < 20:

            ess, _ = calculate_ess(signal)

            ers = (
                0.30 * sqs +
                0.30 * confidence +
                0.40 * ess
            )

        else:

            ess = np.nan
            ers = np.nan

        rows.append({
            "Heartbeat": index + 1,
            "R_Peak_Sample": positions[index],
            "Prediction": prediction,
            "Confidence": confidence,
            "SQS": sqs,
            "ESS": ess,
            "ERS": ers
        })

        progress.progress(
            (index + 1) / len(beats)
        )

    return pd.DataFrame(rows)


# ==================================================
# 18. PDF RECORDING REPORT
# ==================================================

def generate_recording_pdf(results, title):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 15),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        ),
        Spacer(1, 15)
    ]

    valid = results[results["Prediction"] != "Error"]

    summary = [
        ["Metric", "Value"],
        ["Heartbeats", str(len(valid))],
        ["Average confidence",
         f"{valid['Confidence'].mean() * 100:.2f}%"],
        ["Average SQS",
         f"{valid['SQS'].mean():.4f}"]
    ]

    table = Table(summary, colWidths=[220, 220])

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#1e3a5f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    elements.append(table)

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "Heartbeat results are available in the accompanying CSV file. "
            "This prototype is not clinically validated.",
            styles["Normal"]
        )
    )

    document.build(elements)

    buffer.seek(0)

    return buffer


# ==================================================
# 19. SIDEBAR NAVIGATION
# ==================================================

page = st.sidebar.radio(
    "Choose Analysis",
    [
        "Single ECG Analysis",
        "Batch ECG Analysis",
        "Full ECG Recording Analysis",
        "About the Project"
    ]
)


# ==================================================
# 20. SINGLE ECG ANALYSIS
# ==================================================

if page == "Single ECG Analysis":

    st.header("Single Heartbeat Analysis")

    uploaded_file = st.file_uploader(
        "Upload preprocessed ECG heartbeat CSV",
        type=["csv"],
        key="single_upload"
    )

    if uploaded_file:

        try:

            signal = read_heartbeat_csv(uploaded_file)

            result = analyse_heartbeat(signal)

            st.success("ECG analysed successfully")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Class",
                result["prediction"]
            )

            col2.metric(
                "CNN Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            col3.metric(
                "Signal Quality",
                f"{result['sqs']:.3f}"
            )

            st.info(
                CLASS_DESCRIPTIONS.get(
                    result["prediction"],
                    "No description available"
                )
            )

            st.subheader("ECG Waveform")

            waveform_fig = create_waveform_plot(signal)

            st.pyplot(waveform_fig)

            st.subheader("Class Confidence")

            probability_df = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Confidence": result["probabilities"]
            })

            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.subheader("Model Comparison")

            comparison_df = pd.DataFrame({
                "Model": [
                    "CNN",
                    "Random Forest",
                    "XGBoost"
                ],
                "Prediction": [
                    result["prediction"],
                    result["rf"],
                    result["xgb"]
                ]
            })

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Reliability Assessment")

            col1, col2 = st.columns(2)

            col1.metric("ESS", f"{result['ess']:.4f}")

            col2.metric(
                "ERS",
                f"{result['ers'] * 100:.2f}%"
            )

            gauge_fig = create_ers_gauge(result["ers"])

            st.pyplot(gauge_fig)

            st.subheader("Model Explanation")

            explanation_fig = create_explanation_plot(
                signal,
                result["attribution"]
            )

            st.pyplot(explanation_fig)

            report_df = pd.DataFrame([{
                "Predicted_Class": result["prediction"],
                "Confidence": result["confidence"],
                "SQS": result["sqs"],
                "ESS": result["ess"],
                "ERS": result["ers"],
                "Random_Forest": result["rf"],
                "XGBoost": result["xgb"]
            }])

            st.download_button(
                "Download CSV Report",
                report_df.to_csv(index=False),
                file_name="ecg_analysis_report.csv",
                mime="text/csv"
            )

            pdf = generate_pdf_report(
                result,
                waveform_fig
            )

            st.download_button(
                "Download PDF Report",
                pdf,
                file_name="ECG_Trust_AI_Report.pdf",
                mime="application/pdf"
            )

            plt.close(waveform_fig)
            plt.close(gauge_fig)
            plt.close(explanation_fig)

        except Exception as e:

            st.error(f"Analysis error: {e}")


# ==================================================
# 21. BATCH HEARTBEAT ANALYSIS
# ==================================================

elif page == "Batch ECG Analysis":

    st.header("Batch ECG Heartbeat Analysis")

    uploaded_files = st.file_uploader(
        "Upload multiple 252-sample CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if uploaded_files:

        if len(uploaded_files) > 30:

            st.warning("Maximum 30 files per batch")

        elif st.button("Analyse All ECG Files"):

            rows = []

            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):

                try:

                    signal = read_heartbeat_csv(uploaded_file)

                    result = analyse_heartbeat(signal)

                    rows.append({
                        "File": uploaded_file.name,
                        "Prediction": result["prediction"],
                        "Confidence": result["confidence"],
                        "SQS": result["sqs"],
                        "ESS": result["ess"],
                        "ERS": result["ers"],
                        "CNN": result["prediction"],
                        "Random Forest": result["rf"],
                        "XGBoost": result["xgb"]
                    })

                except Exception as e:

                    rows.append({
                        "File": uploaded_file.name,
                        "Prediction": "Error",
                        "Confidence": np.nan,
                        "SQS": np.nan,
                        "ESS": np.nan,
                        "ERS": np.nan,
                        "CNN": str(e),
                        "Random Forest": "Unavailable",
                        "XGBoost": "Unavailable"
                    })

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.session_state["batch_results"] = pd.DataFrame(rows)

        if "batch_results" in st.session_state:

            batch_df = st.session_state["batch_results"]

            st.subheader("Batch Results")

            st.dataframe(
                batch_df,
                use_container_width=True,
                hide_index=True
            )

            valid = batch_df[
                batch_df["Prediction"] != "Error"
            ]

            if not valid.empty:

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Heartbeats Analysed",
                    len(valid)
                )

                col2.metric(
                    "Average Confidence",
                    f"{valid['Confidence'].mean() * 100:.2f}%"
                )

                col3.metric(
                    "Average ERS",
                    f"{valid['ERS'].mean() * 100:.2f}%"
                )

                st.subheader("Class Distribution")

                st.bar_chart(
                    valid["Prediction"].value_counts()
                )

                st.download_button(
                    "Download Batch CSV",
                    batch_df.to_csv(index=False),
                    file_name="ECG_Batch_Analysis.csv",
                    mime="text/csv"
                )


# ==================================================
# 22. FULL ECG RECORDING ANALYSIS
# ==================================================

elif page == "Full ECG Recording Analysis":

    st.header("Full ECG Recording Analysis")

    recording_type = st.radio(
        "Select recording format",
        [
            "CSV Recording",
            "MIT-BIH WFDB Recording"
        ]
    )

    if recording_type == "CSV Recording":

        uploaded_file = st.file_uploader(
            "Upload complete ECG CSV",
            type=["csv"],
            key="recording_csv"
        )

        if uploaded_file:

            try:

                preview = pd.read_csv(uploaded_file)

                columns = list(preview.columns)

                selected_column = st.selectbox(
                    "Select ECG signal column",
                    columns
                )

                fs = st.number_input(
                    "Sampling rate (Hz)",
                    min_value=50,
                    max_value=2000,
                    value=360
                )

                if st.button("Analyse Full Recording"):

                    uploaded_file.seek(0)

                    beats, positions, fs, method = (
                        load_csv_recording(
                            uploaded_file,
                            selected_column,
                            fs
                        )
                    )

                    if len(beats) == 0:

                        st.error("No heartbeats detected")

                    else:

                        results = analyse_recording(
                            beats,
                            positions
                        )

                        st.success(
                            f"{len(beats)} heartbeats extracted"
                        )

                        st.info(
                            f"Detection method: {method}"
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        st.subheader("Recording Summary")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Heartbeats",
                            len(results)
                        )

                        col2.metric(
                            "Average Confidence",
                            f"{results['Confidence'].mean() * 100:.2f}%"
                        )

                        col3.metric(
                            "Average SQS",
                            f"{results['SQS'].mean():.3f}"
                        )

                        st.subheader("Class Distribution")

                        st.bar_chart(
                            results["Prediction"].value_counts()
                        )

                        st.download_button(
                            "Download Recording CSV",
                            results.to_csv(index=False),
                            file_name="full_ecg_analysis.csv",
                            mime="text/csv"
                        )

                        pdf = generate_recording_pdf(
                            results,
                            "Full ECG Recording Report"
                        )

                        st.download_button(
                            "Download Recording PDF",
                            pdf,
                            file_name="full_ecg_report.pdf",
                            mime="application/pdf"
                        )

            except Exception as e:

                st.error(f"Recording analysis failed: {e}")

    else:

        uploaded_files = st.file_uploader(
            "Upload WFDB files (.hea, .dat and optional .atr)",
            type=["hea", "dat", "atr"],
            accept_multiple_files=True,
            key="wfdb_upload"
        )

        if uploaded_files:

            st.info(
                "Upload matching .hea and .dat files. "
                "The .atr annotation file is recommended."
            )

            if st.button("Analyse WFDB Recording"):

                try:

                    beats, positions, fs, method = (
                        load_wfdb_recording(uploaded_files)
                    )

                    if len(beats) == 0:

                        st.error("No valid heartbeats extracted")

                    else:

                        results = analyse_recording(
                            beats,
                            positions
                        )

                        st.success(
                            f"{len(beats)} heartbeats extracted"
                        )

                        st.info(
                            f"Sampling rate: {fs} Hz | Method: {method}"
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        st.subheader("Recording Summary")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Heartbeats",
                            len(results)
                        )

                        col2.metric(
                            "Average Confidence",
                            f"{results['Confidence'].mean() * 100:.2f}%"
                        )

                        col3.metric(
                            "Average SQS",
                            f"{results['SQS'].mean():.3f}"
                        )

                        st.subheader("Class Distribution")

                        st.bar_chart(
                            results["Prediction"].value_counts()
                        )

                        st.download_button(
                            "Download WFDB CSV Report",
                            results.to_csv(index=False),
                            file_name="MITBIH_recording_results.csv",
                            mime="text/csv"
                        )

                        pdf = generate_recording_pdf(
                            results,
                            "MIT-BIH ECG Recording Report"
                        )

                        st.download_button(
                            "Download WFDB PDF Report",
                            pdf,
                            file_name="MITBIH_recording_report.pdf",
                            mime="application/pdf"
                        )

                except Exception as e:

                    st.error(f"WFDB analysis failed: {e}")


# ==================================================
# 23. ABOUT THE PROJECT
# ==================================================

elif page == "About the Project":

    st.header("About ECG-Trust AI")

    st.markdown("""
    ### Research Objective

    To investigate ECG heartbeat classification and the stability
    of AI-generated explanations under signal perturbations.

    ### Main Components

    **1. Classification**
    - 1D CNN
    - Random Forest
    - XGBoost

    **2. Signal Quality Score (SQS)**

    Estimates signal quality using signal characteristics.

    **3. Explanation Stability Score (ESS)**

    Compares Integrated Gradients explanations for clean and
    synthetically perturbed heartbeat signals.

    **4. ECG Reliability Score (ERS)**

    Combines signal quality, CNN confidence and explanation stability.

    ### ERS Formula

    ERS = 0.30 × SQS + 0.30 × Confidence + 0.40 × ESS

    ### Dataset

    MIT-BIH Arrhythmia Database.

    ### Limitations

    - Raw recording automatic R-peak detection is experimental
    - ESS and ERS are research metrics, not clinical measurements
    - Recording-level ESS and ERS are calculated only for the first
      20 extracted beats in this prototype
    - Full raw ECG preprocessing requires further validation
    - This is not a medical diagnostic device
    """)


# ==================================================
# 24. FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "ECG-Trust AI | Academic Research Prototype | "
    "Not intended for clinical diagnosis"
)
