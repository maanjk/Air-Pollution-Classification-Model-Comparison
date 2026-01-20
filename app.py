import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load models & feature list
# --------------------------
@st.cache_resource
def load_artifacts():
    svm = joblib.load("air_svm_model.pkl")
    logreg = joblib.load("air_logreg_model.pkl")
    dt = joblib.load("air_dt_model.pkl")
    features = joblib.load("air_feature_names.pkl")
    return svm, logreg, dt, features

svm_model, logreg_model, dt_model, feature_names = load_artifacts()

st.set_page_config(page_title="Air Pollution – SVM/LogReg/DT", layout="centered")

st.title("Air Pollution Classification – Model Comparison")
st.write(
    "This app uses three models (SVM, Logistic Regression, Decision Tree) "
    "trained on the China Air Pollution dataset to predict whether pollution is **high** (1) or **low** (0).\n\n"
    "**Note:** This is an educational demo, not an official air‑quality tool."
)

st.markdown("---")
st.header("Enter Sensor Values")

# --------------------------
# Input form
# --------------------------
cols = st.columns(2)
user_data = {}

for i, feat in enumerate(feature_names):
    with cols[i % 2]:
        # simple numeric input; default 0.0
        user_data[feat] = st.number_input(f"{feat}", value=0.0)

input_df = pd.DataFrame([user_data], columns=feature_names)

st.markdown("---")

if st.button("Predict with All Models"):
    # Predict probabilities (for class 1 = high pollution)
    proba_svm = svm_model.predict_proba(input_df)[0, 1]
    proba_lr = logreg_model.predict_proba(input_df)[0, 1]
    proba_dt = dt_model.predict_proba(input_df)[0, 1]

    pred_svm = int(proba_svm >= 0.5)
    pred_lr = int(proba_lr >= 0.5)
    pred_dt = int(proba_dt >= 0.5)

    st.subheader("Predictions")
    st.write(f"**SVM**: class = {pred_svm},  probability(high) = {proba_svm:.3f}")
    st.write(f"**Logistic Regression**: class = {pred_lr},  probability(high) = {proba_lr:.3f}")
    st.write(f"**Decision Tree**: class = {pred_dt},  probability(high) = {proba_dt:.3f}")

    # Bar chart of probabilities
    st.subheader("Model Comparison (Probability of High Pollution)")
    prob_df = pd.DataFrame({
        "Model": ["SVM", "Logistic Regression", "Decision Tree"],
        "Probability High": [proba_svm, proba_lr, proba_dt],
    }).set_index("Model")

    st.bar_chart(prob_df)

    st.caption(
        "Probabilities are for the 'high pollution' class (1). "
        "Bars closer to 1.0 indicate stronger confidence in high pollution."
    )