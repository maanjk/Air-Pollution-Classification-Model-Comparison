# Air Pollution Classification & Model Comparison

This project classifies air pollution levels as **High** or **Low** using sensor data from China (2015-2025). It trains and compares three machine learning models: **Support Vector Machine (SVM)**, **Logistic Regression**, and **Decision Tree**.

## 🚀 Live Demo
[Click here to view the Streamlit App]((https://air-pollution-classification-model-comparison.streamlit.app/))  
*(Replace the link above with your actual deployed app link)*

## 📊 Models Used
*   **SVM (Support Vector Machine):** Good for high-dimensional boundaries.
*   **Logistic Regression:** A baseline linear classifier.
*   **Decision Tree:** Provides interpretable decision rules.

## 🛠️ Tech Stack
*   **Python** (Pandas, NumPy, Scikit-learn)
*   **Streamlit** (Web Interface)
*   **Joblib** (Model persistence)

## 📂 Project Structure
*   `app.py`: Main Streamlit application.
*   `*.pkl`: Trained model files.
*   `requirements.txt`: Python dependencies.

## 🏃‍♂️ How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
