import streamlit as st
import pandas as pd
import joblib

# --------------------------
# 1. Page Configuration
# (Must be the first st command)
# --------------------------
st.set_page_config(
    page_title="AirGuard - Pollution AI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# 2. Load models & features
# --------------------------
@st.cache_resource
def load_artifacts():
    # Load the files saved from Kaggle
    svm = joblib.load("air_svm_model.pkl")
    logreg = joblib.load("air_logreg_model.pkl")
    dt = joblib.load("air_dt_model.pkl")
    features = joblib.load("air_feature_names.pkl")
    return svm, logreg, dt, features

try:
    svm_model, logreg_model, dt_model, feature_names = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading models. Please ensure .pkl files are in the same folder.\n\nDetails: {e}")
    st.stop()

# --------------------------
# 3. Sidebar - User Inputs
# --------------------------
with st.sidebar:
    st.header("🎛️ Sensor Configuration")
    st.write("Adjust sensor readings below:")
    
    user_data = {}
    
    # Dynamically create inputs based on the feature names list
    for feat in feature_names:
        unit = "mg/m³" if "CO" in feat.upper() else "µg/m³"
        user_data[feat] = st.number_input(
            f"{feat} ({unit})", 
            value=0.0, 
            step=0.1,
            format="%.2f"
        )
    
    st.markdown("---")
    st.button("Reset Values", type="secondary")

# --------------------------
# 4. Main Dashboard UI
# --------------------------
st.title("🏭 Air Pollution Classifier")
st.markdown("""
This dashboard compares **SVM**, **Logistic Regression**, and **Decision Tree** models 
to detect **High Pollution Events** based on sensor data.
""")

# Create a container for results
result_container = st.container()

if st.button("🚀 Analyze Air Quality", type="primary", use_container_width=True):
    
    # Prepare input dataframe
    input_df = pd.DataFrame([user_data], columns=feature_names)

    # Get Probabilities (Class 1 = High Pollution)
    proba_svm = svm_model.predict_proba(input_df)[0, 1]
    proba_lr = logreg_model.predict_proba(input_df)[0, 1]
    proba_dt = dt_model.predict_proba(input_df)[0, 1]

    with result_container:
        st.divider()
        st.subheader("🔍 Model Consensus")
        
        # Display cards for each model
        col1, col2, col3 = st.columns(3)

        # Helper function to render a result card
        def display_card(col, model_name, prob):
            with col:
                is_high = prob >= 0.5
                status_text = "HIGH POLLUTION" if is_high else "LOW POLLUTION"
                
                st.markdown(f"**{model_name}**")
                
                if is_high:
                    st.error(f"🔴 {status_text}")
                else:
                    st.success(f"🟢 {status_text}")
                
                # Progress bar for confidence
                st.progress(prob, text=f"Risk Confidence: {prob:.1%}")

        display_card(col1, "Support Vector Machine", proba_svm)
        display_card(col2, "Logistic Regression", proba_lr)
        display_card(col3, "Decision Tree", proba_dt)

        # --------------------------
        # 5. Comparison Chart
        # --------------------------
        st.divider()
        st.subheader("📊 Probability Comparison")
        
        chart_data = pd.DataFrame({
            "Model": ["SVM", "Logistic Regression", "Decision Tree"],
            "Risk Probability": [proba_svm, proba_lr, proba_dt]
        })

        st.bar_chart(
            chart_data, 
            x="Model", 
            y="Risk Probability", 
            color="#FF4B4B",  # Red color to signify risk
            use_container_width=True
        )

else:
    # Initial state helper message
    st.info("👈 Enter sensor data in the sidebar and click 'Analyze Air Quality' to see results.")

# --------------------------
# Footer
# --------------------------
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    *   **Data:** Trained on China Air Pollution dataset.
    *   **Logic:** If the predicted probability is > 50%, it is classified as **High Pollution**.
    *   **Models:**
        *   *SVM:* Finds the best boundary between high/low air quality.
        *   *LogReg:* Estimates probability using a linear equation.
        *   *Decision Tree:* Uses a flowchart-like structure to decide.
    """)
