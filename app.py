
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="HealthPredict AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------- FUTURISTIC THEME ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(circle at top left, #172b48, #080d1b 55%);
    color: #f4f7ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #101a2d;
    border-right: 1px solid #263b56;
}

h1 {
    color: #45e6e0 !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #e8f3ff !important;
}

p, label {
    color: #c1cce0;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17263e, #10192c);
    border: 1px solid #28506a;
    padding: 20px;
    border-radius: 16px;
}

[data-testid="stMetricValue"] {
    color: #45e6e0 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #0ea5a5, #6655d9);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 45px;
    font-weight: 700;
}

.stButton > button:hover {
    box-shadow: 0 0 18px #45e6e055;
    color: white;
}

[data-testid="stChatMessage"] {
    background: #142239;
    border: 1px solid #293d59;
    border-radius: 15px;
}

hr {
    border-color: #263b56;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("health_predict_model.keras")
    scaler = joblib.load("health_scaler.pkl")
    features = joblib.load("health_features.pkl")
    return model, scaler, features


try:
    model, scaler, features = load_resources()
except Exception as e:
    st.error("Unable to load model files.")
    st.write("Check whether all three model files are uploaded to GitHub.")
    st.code(str(e))
    st.stop()


# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "chat" not in st.session_state:
    st.session_state.chat = []


# ---------------- SIDEBAR ----------------

st.sidebar.title("🩺 HealthPredict AI")
st.sidebar.caption("Intelligent Health Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Diabetes Prediction",
        "Health Assistant",
        "Prediction History",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.caption("AI Healthcare Platform | Version 2.0")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("HealthPredict AI")
    st.subheader("Next Generation Educational Health Analytics")

    st.write(
        "Explore an Artificial Intelligence-based diabetes "
        "prediction demonstration through an interactive dashboard."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Technology", "ANN")
    col2.metric("Dataset", "Pima Indians")
    col3.metric("Platform", "Streamlit")

    st.divider()

    st.subheader("Platform Features")

    c1, c2 = st.columns(2)

    with c1:
        st.info("🔍 **Diabetes Prediction**\n\nExplore model predictions using health measurements.")

        st.info("💬 **Health Assistant**\n\nLearn general information about diabetes.")

    with c2:
        st.info("📊 **Prediction History**\n\nReview your current session's predictions.")

        st.info("🧠 **Neural Network**\n\nPowered by a trained ANN model.")

    st.warning(
        "Educational demonstration only. "
        "This application is not a medical diagnostic tool."
    )


# ---------------- DIABETES PREDICTION ----------------

elif page == "Diabetes Prediction":

    st.title("Diabetes Risk Prediction")
    st.write("Enter the measurements below.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies", 0, 20, 1
            )

            glucose = st.number_input(
                "Glucose", 0, 300, 120
            )

            blood_pressure = st.number_input(
                "Blood Pressure", 0, 200, 70
            )

            skin_thickness = st.number_input(
                "Skin Thickness", 0, 100, 20
            )

        with col2:
            insulin = st.number_input(
                "Insulin", 0, 900, 79
            )

            bmi = st.number_input(
                "BMI", 0.0, 100.0, 25.0
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                0.0, 3.0, 0.5
            )

            age = st.number_input(
                "Age", 1, 120, 30
            )

        submitted = st.form_submit_button(
            "Analyze Health Data",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age
            ]],
            columns=features
        )

        try:
            scaled_data = scaler.transform(input_data)

            score = float(
                model.predict(
                    scaled_data,
                    verbose=0
                )[0][0]
            )

            prediction = score >= 0.5

            st.divider()
            st.subheader("AI Analysis Result")

            if prediction:
                st.error("Higher predicted risk category")
            else:
                st.success("Lower predicted risk category")

            st.metric(
                "Model Output Score",
                f"{score * 100:.2f}%"
            )

            st.progress(
                min(max(score, 0.0), 1.0)
            )

            st.caption(
                "This is a model output score, not an individual's "
                "actual medical probability."
            )

            st.session_state.history.append({
                "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Result": "Higher predicted risk" if prediction else "Lower predicted risk",
                "Score": round(score * 100, 2)
            })

            st.warning(
                "This prediction is not a diagnosis. "
                "Consult a qualified healthcare professional "
                "for medical testing and advice."
            )

        except Exception as e:
            st.error("Prediction error")
            st.code(str(e))


# ---------------- HEALTH ASSISTANT ----------------

elif page == "Health Assistant":

    st.title("AI Health Assistant")

    st.caption(
        "Educational rule-based assistant. Not a medical diagnosis service."
    )

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask about diabetes, BMI, insulin or healthy habits..."
    )

    if question:

        st.session_state.chat.append({
            "role": "user",
            "content": question
        })

        q = question.lower()

        if "symptom" in q:
            answer = (
                "Common diabetes symptoms can include increased thirst, "
                "frequent urination, tiredness and blurred vision. "
                "Some people have no noticeable symptoms."
            )

        elif any(word in q for word in ["food", "diet", "eat"]):
            answer = (
                "A balanced diet can include vegetables, whole grains, "
                "pulses and appropriate protein sources. "
                "Individual needs vary."
            )

        elif any(word in q for word in ["exercise", "walking", "workout"]):
            answer = (
                "Regular physical activity can support overall health. "
                "Walking and cycling are examples of physical activities."
            )

        elif "insulin" in q:
            answer = (
                "Insulin is a hormone that helps glucose enter body cells "
                "and regulates blood sugar levels."
            )

        elif "bmi" in q:
            answer = (
                "BMI stands for Body Mass Index. It is calculated using "
                "weight and height and is a screening measure."
            )

        elif any(word in q for word in ["hello", "hi", "hey"]):
            answer = (
                "Hello! Welcome to HealthPredict AI. "
                "Ask me general questions about diabetes and healthy habits."
            )

        else:
            answer = (
                "I can provide general educational information about "
                "diabetes, symptoms, BMI, insulin, food and exercise. "
                "Please consult a healthcare professional for personal medical advice."
            )

        st.session_state.chat.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear Conversation"):
        st.session_state.chat = []
        st.rerun()


# ---------------- PREDICTION HISTORY ----------------

elif page == "Prediction History":

    st.title("Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download CSV Report",
            data=csv_data,
            file_name="healthpredict_history.csv",
            mime="text/csv"
        )

    else:
        st.info("No predictions available in this session yet.")


# ---------------- ABOUT ----------------

elif page == "About":

    st.title("About HealthPredict AI")

    st.write("""
    HealthPredict AI is an educational Artificial Intelligence project
    demonstrating diabetes risk prediction using an Artificial Neural Network.

    **Technologies Used**
    - Python
    - TensorFlow / Keras
    - Scikit-learn
    - Pandas
    - NumPy
    - Streamlit

    **Dataset:** Pima Indians Diabetes Dataset.

    **Model:** Artificial Neural Network (ANN).

    **Purpose:** AI/ML learning and demonstration.
    """)

    st.warning(
        "This application is not clinically validated. "
        "It must not be used for diagnosis or treatment decisions. "
        "Consult a qualified healthcare professional."
    )
