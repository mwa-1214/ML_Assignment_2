
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(
    page_title="Electrical Fault Detection",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electrical Fault Detection")
st.write(
    "Enter the three-phase current and voltage measurements "
    "to compare Set Q Logistic Regression and RBF SVM."
)

FEATURES_R = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
FEATURES_Q = FEATURES_R + [
    "Ia_sq", "Ib_sq", "Ic_sq",
    "Va_sq", "Vb_sq", "Vc_sq"
]


@st.cache_resource
def load_models():
    theta_q = np.load(RESULTS_DIR / "theta_Q.npy")
    rbf_svm = joblib.load(RESULTS_DIR / "best_rbf_svm.pkl")

    mean_r = pd.read_csv(
        RESULTS_DIR / "scaler_R_mean.csv", index_col=0
    ).iloc[:, 0].reindex(FEATURES_R).to_numpy()

    std_r = pd.read_csv(
        RESULTS_DIR / "scaler_R_std.csv", index_col=0
    ).iloc[:, 0].reindex(FEATURES_R).to_numpy()

    mean_q = pd.read_csv(
        RESULTS_DIR / "scaler_Q_mean.csv", index_col=0
    ).iloc[:, 0].reindex(FEATURES_Q).to_numpy()

    std_q = pd.read_csv(
        RESULTS_DIR / "scaler_Q_std.csv", index_col=0
    ).iloc[:, 0].reindex(FEATURES_Q).to_numpy()

    return theta_q, rbf_svm, mean_r, std_r, mean_q, std_q


theta_q, rbf_svm, mean_r, std_r, mean_q, std_q = load_models()

st.subheader("Electrical Measurements")

with st.form("fault_inputs"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Ia = st.number_input("Ia", value=0.0, format="%.6f")
        Ib = st.number_input("Ib", value=0.0, format="%.6f")

    with col2:
        Ic = st.number_input("Ic", value=0.0, format="%.6f")
        Va = st.number_input("Va", value=0.0, format="%.6f")

    with col3:
        Vb = st.number_input("Vb", value=0.0, format="%.6f")
        Vc = st.number_input("Vc", value=0.0, format="%.6f")

    submitted = st.form_submit_button(
        "Predict Fault", type="primary"
    )


if submitted:
    raw_r = np.array([Ia, Ib, Ic, Va, Vb, Vc], dtype=float)

    # Standardize Set R features using training statistics.
    scaled_r = (raw_r - mean_r) / std_r

    # Create squared features from raw values, then standardize.
    raw_q = np.concatenate([raw_r, raw_r ** 2])
    scaled_q = (raw_q - mean_q) / std_q

    # Set Q Logistic Regression
    x_logistic = np.concatenate([[1.0], scaled_q])
    z = float(np.dot(x_logistic, theta_q))

    if z >= 0:
        probability = 1.0 / (1.0 + np.exp(-z))
    else:
        exp_z = np.exp(z)
        probability = exp_z / (1.0 + exp_z)

    logistic_prediction = int(probability >= 0.5)

    # RBF SVM
    svm_prediction = int(
        rbf_svm.predict(scaled_r.reshape(1, -1))[0]
    )

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Set Q Logistic Regression")
        st.metric("Fault probability", f"{probability * 100:.2f}%")
        if logistic_prediction == 1:
            st.error("Prediction: FAULT")
        else:
            st.success("Prediction: NO FAULT")

    with col2:
        st.markdown("### RBF SVM")
        if svm_prediction == 1:
            st.error("Prediction: FAULT")
        else:
            st.success("Prediction: NO FAULT")

    if logistic_prediction == svm_prediction:
        st.info("Both models agree.")
    else:
        st.warning(
            "The models disagree. Further investigation may be needed."
        )

    st.caption(
        "This is an educational demonstration, not a certified "
        "electrical protection system."
    )
