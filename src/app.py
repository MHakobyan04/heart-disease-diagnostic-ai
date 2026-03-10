import streamlit as st
import pandas as pd
import joblib
import os

# Path Configuration
# Setting up paths relative to the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Page Configuration
st.set_page_config(
    page_title="Heart Disease AI Diagnostic",
    page_icon="🩺",
    layout="wide"
)


# Asset Loading (Cached)
@st.cache_resource
def load_ml_assets():

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        return None, None


model, scaler = load_ml_assets()

if model is None or scaler is None:
    st.error("Missing Assets: Ensure that 'models/trained_model.pkl' and 'models/scaler.pkl' exist.")
    st.stop()

# Header Section
st.title("🩺 Heart Disease AI Diagnostic System")
st.markdown("""
This diagnostic tool utilizes a **Logistic Regression** model trained on clinical data 
to predict the likelihood of heart disease. Please input the patient metrics below.
""")

st.divider()

# User Input Form
with st.form("diagnostic_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Vitals")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

    with col2:
        st.subheader("Clinical Observations")
        cp = st.selectbox("Chest Pain Type (CP)", options=[0, 1, 2, 3],
                          help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                           format_func=lambda x: "True" if x == 1 else "False")
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
        ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Form Submission
    submit_button = st.form_submit_button("Perform Diagnostic", type="primary")

# Inference Logic
if submit_button:
    # 1. Feature aggregation into DataFrame
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=features)

    # 2. Data Transformation (Scaling)
    # Using the scaler fitted during the preprocessing stage
    scaled_input = scaler.transform(input_df)

    # 3. Model Prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Results Visualization
    st.divider()
    if prediction == 1:
        st.error(f"### High Risk: Heart Disease Likely")
        st.write(f"The model predicts a heart disease risk of **{probability * 100:.2f}%**.")
    else:
        st.success(f"### Low Risk: Heart Disease Unlikely")
        st.write(
            f"The model predicts a healthy cardiac state with a probability of **{(1 - probability) * 100:.2f}%**.")

    st.warning(
        "**Medical Disclaimer:** This application is an AI demonstration and should not be used as a primary diagnostic tool.")