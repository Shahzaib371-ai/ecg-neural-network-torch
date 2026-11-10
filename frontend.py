
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from task2 import ECGNet

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "results" / "best_ecgnet.pth"
TEST_PATH = BASE_DIR / "data" / "mitbih_test.csv"

CLASS_NAMES = [
    "N: Normal beat",
    "S: Supraventricular ectopic",
    "V: Ventricular ectopic",
    "F: Fusion beat",
    "Q: Unknown / Paced",
]

st.set_page_config(
    page_title="ECG Heartbeat Classifier",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
}
[data-testid="stMetric"] {
    background: #172235;
    border: 1px solid #29384d;
    padding: 16px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.is_file():
        return None, f"Model not found: {MODEL_PATH}"

    try:
        model = ECGNet()
        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model, None

    except TypeError:
        try:
            model = ECGNet()
            state_dict = torch.load(
                MODEL_PATH,
                map_location="cpu"
            )
            model.load_state_dict(state_dict)
            model.eval()
            return model, None
        except Exception as e:
            return None, str(e)

    except Exception as e:
        return None, str(e)


@st.cache_data
def load_test_data():
    if not TEST_PATH.is_file():
        return None
    return pd.read_csv(TEST_PATH, header=None)


def validate_signal(values):
    signal = np.asarray(values, dtype=np.float32).reshape(-1)

    if len(signal) != 187:
        raise ValueError("ECG must contain exactly 187 samples.")

    if not np.isfinite(signal).all():
        raise ValueError("Invalid ECG values detected.")

    return signal


def predict(model, signal):
    x = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = int(torch.argmax(probabilities).item())

    return prediction, probabilities.numpy()


# SIDEBAR
st.sidebar.title("🫀 ECG AI SYSTEM")
st.sidebar.caption("Heartbeat Classification")
st.sidebar.divider()

input_method = st.sidebar.radio(
    "Select Input Source",
    ["Select Test Dataset Row", "Upload ECG CSV"]
)

st.sidebar.divider()
st.sidebar.write("Architecture: 187 → 128 → 64 → 5")
st.sidebar.write("Framework: PyTorch")
st.sidebar.write("Sampling Rate: 125 Hz")


# HEADER
st.title("🫀 ECG Heartbeat Classification")
st.caption("AI-powered ECG Signal Analysis Dashboard")
st.divider()


# LOAD MODEL
model, error = load_trained_model()

if model is None:
    st.error(f"Model loading failed: {error}")
    st.stop()

st.sidebar.success("Trained Model Loaded!")


# INPUT DATA
signal = None
true_label = None

if input_method == "Select Test Dataset Row":

    test_df = load_test_data()

    if test_df is None:
        st.error("data/mitbih_test.csv not found.")

    elif test_df.shape[1] < 187:
        st.error("Dataset must contain at least 187 columns.")

    else:
        row_index = st.sidebar.number_input(
            "Select Row Index",
            min_value=0,
            max_value=len(test_df) - 1,
            value=0,
            step=1
        )

        row = test_df.iloc[int(row_index)]

        try:
            signal = validate_signal(row.iloc[:187].values)

            if len(row) > 187 and pd.notna(row.iloc[187]):
                label = int(row.iloc[187])
                if 0 <= label < 5:
                    true_label = label

        except Exception as e:
            st.error(f"Invalid row: {e}")

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload ECG CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(
                uploaded_file,
                header=None
            )

            if uploaded_df.empty or uploaded_df.shape[1] < 187:
                raise ValueError("CSV needs at least 187 values.")

            signal = validate_signal(
                pd.to_numeric(
                    uploaded_df.iloc[0, :187],
                    errors="raise"
                ).values
            )

            if uploaded_df.shape[1] > 187:
                label = pd.to_numeric(
                    pd.Series([uploaded_df.iloc[0, 187]]),
                    errors="coerce"
                ).iloc[0]

                if pd.notna(label) and 0 <= int(label) < 5:
                    true_label = int(label)

        except Exception as e:
            st.error(f"Upload error: {e}")


# DASHBOARD
if signal is not None:

    prediction, probabilities = predict(model, signal)

    confidence = probabilities[prediction] * 100

    st.subheader("Prediction Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Class",
        CLASS_NAMES[prediction].split(":")[0]
    )

    col2.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    col3.metric(
        "Total Samples",
        "187"
    )

    if true_label is not None:
        st.write(f"Actual Class: {CLASS_NAMES[true_label]}")

        if prediction == true_label:
            st.success("Correct Prediction!")
        else:
            st.error("Incorrect Prediction!")

    st.divider()

    left, right = st.columns([1.5, 1])

    # ECG GRAPH
    with left:

        st.subheader("📈 ECG Signal Waveform")

        time_ms = np.arange(187) * 1000 / 125

        fig, ax = plt.subplots(figsize=(9, 4))

        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")

        ax.plot(
            time_ms,
            signal,
            color="#38bdf8",
            linewidth=1.6
        )

        ax.set_title("ECG Beat Signal", color="white")
        ax.set_xlabel("Time (ms)", color="white")
        ax.set_ylabel("Normalized Amplitude", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # PROBABILITY CHART
    with right:

        st.subheader("🎯 Class Probabilities")

        probability_df = pd.DataFrame({
            "Class": [
                name.split(":")[0]
                for name in CLASS_NAMES
            ],
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Class"),
            color="#38bdf8"
        )

        for name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):
            st.write(
                f"**{name}:** {probability * 100:.2f}%"
            )

    st.divider()

    st.subheader("Detailed Results")

    results_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability (%)": np.round(
            probabilities * 100,
            3
        )
    })

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Educational project only. Not a medical diagnosis."
    )

else:
    st.info(
        "Select a test row or upload an ECG CSV "
        "from the sidebar to start."
    )

st.divider()
st.caption("ECG Heartbeat Classifier | Custom PyTorch Neural Network")
