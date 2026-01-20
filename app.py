import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# 1. Page Config (Dark Mode friendly)
# --------------------------
st.set_page_config(
    page_title="AirGuard Pro",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# 2. Load Models
# --------------------------
@st.cache_resource
def load_artifacts():
    svm = joblib.load("air_svm_model.pkl")
    logreg = joblib.load("air_logreg_model.pkl")
    dt = joblib.load("air_dt_model.pkl")
    features = joblib.load("air_feature_names.pkl")
    return svm, logreg, dt, features

try:
    svm_model, logreg_model, dt_model, feature_names = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading models. Ensure .pkl files are in the folder. {e}")
    st.stop()

# --------------------------
# 3. Sidebar Inputs
# --------------------------
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.write("Simulate sensor readings:")
    
    user_data = {}
    for feat in feature_names:
        unit = "mg/m³" if "CO" in feat.upper() else "µg/m³"
        user_data[feat] = st.number_input(
            f"{feat} ({unit})", 
            value=0.0, 
            step=1.0,
            format="%.2f"
        )
    
    st.markdown("---")
    st.caption("Adjust values to see how the models react in real-time.")

# --------------------------
# 4. Main App Logic
# --------------------------
st.title("🌪️ Air Pollution AI Dashboard")

# Create Tabs for a cleaner look
tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "📈 Model Analysis (Line Graphs)"])

# Prepare Input Data
input_df = pd.DataFrame([user_data], columns=feature_names)

# Get Probabilities
prob_svm = svm_model.predict_proba(input_df)[0, 1]
prob_lr = logreg_model.predict_proba(input_df)[0, 1]
prob_dt = dt_model.predict_proba(input_df)[0, 1]

# Average Risk
avg_risk = (prob_svm + prob_lr + prob_dt) / 3

# --- TAB 1: MAIN DASHBOARD ---
with tab1:
    # Top Level Metrics
    st.subheader("Current Air Quality Status")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric(label="Avg Risk Score", value=f"{avg_risk:.1%}", delta="Safe" if avg_risk < 0.5 else "-Danger")
    with col_m2:
        st.metric(label="SVM Confidence", value=f"{prob_svm:.1%}")
    with col_m3:
        st.metric(label="LogReg Confidence", value=f"{prob_lr:.1%}")
    with col_m4:
        st.metric(label="Tree Confidence", value=f"{prob_dt:.1%}")

    st.divider()

    # Layout: Chart on Left, Detailed Cards on Right
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Pollution Profile (Input Shape)")
        # Normalize data for visualization (Area Chart)
        # This shows which pollutant is spiking relative to others
        viz_df = input_df.T.reset_index()
        viz_df.columns = ["Pollutant", "Value"]
        st.area_chart(viz_df.set_index("Pollutant"), color="#00a8cc")
        st.caption("Visual representation of the sensor inputs.")

    with c2:
        st.subheader("Consensus")
        
        models = [("SVM", prob_svm), ("Logistic Regression", prob_lr), ("Decision Tree", prob_dt)]
        
        for name, prob in models:
            is_high = prob >= 0.5
            color = "red" if is_high else "green"
            icon = "🚨" if is_high else "✅"
            
            with st.container():
                st.write(f"**{name}**")
                st.progress(prob)
                if is_high:
                    st.error(f"{icon} HIGH POLLUTION")
                else:
                    st.success(f"{icon} LOW POLLUTION")

# --- TAB 2: LINE GRAPHS (WHAT-IF ANALYSIS) ---
with tab2:
    st.subheader("📈 Sensitivity Analysis (The 'What-If' Graph)")
    st.markdown("How does the risk change if **one** pollutant increases while others stay the same?")
    
    # Let user pick a feature to analyze
    target_feat = st.selectbox("Select Pollutant to Simulate:", feature_names)
    
    # Generate a range of values for this feature (0 to 300)
    x_values = np.linspace(0, 300, 50)
    
    # Create synthetic dataframes for prediction
    sim_data = []
    for x in x_values:
        row = user_data.copy()
        row[target_feat] = x # Override the chosen feature
        sim_data.append(row)
        
    sim_df = pd.DataFrame(sim_data, columns=feature_names)
    
    # Predict on this range
    sim_probs_svm = svm_model.predict_proba(sim_df)[:, 1]
    sim_probs_lr = logreg_model.predict_proba(sim_df)[:, 1]
    
    # Build chart data
    chart_df = pd.DataFrame({
        f"{target_feat} Value": x_values,
        "SVM Risk": sim_probs_svm,
        "LogReg Risk": sim_probs_lr
    }).set_index(f"{target_feat} Value")
    
    # Plot Line Chart
    st.line_chart(chart_df, color=["#FF4B4B", "#0068C9"])
    
    st.info(
        f"**Interpretation:** The graph shows how the probability of 'High Pollution' rises as "
        f"**{target_feat}** increases from 0 to 300. Notice how the SVM (Red) might react differently "
        f"than Logistic Regression (Blue)."
    )
