
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

# ==================================================
# HEALTHPREDICT AI 3.0
# Existing ANN model and scaler are preserved
# ==================================================

st.set_page_config(
    page_title="HealthPredict AI 3.0",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- TRANSLATIONS ----------------

TEXT = {
    "app_name": ("HealthPredict AI", "HealthPredict AI"),
    "tagline": (
        "Your intelligent educational health companion",
        "உங்கள் AI சுகாதாரக் கல்வி உதவியாளர்"
    ),
    "language": ("Language / மொழி", "மொழி / Language"),
    "english": ("English", "English"),
    "tamil": ("தமிழ்", "தமிழ்"),
    "theme": ("Color Theme", "நிறத் தோற்றம்"),
    "navigation": ("Navigation", "பக்கங்கள்"),
    "dashboard": ("Dashboard", "முகப்பு"),
    "prediction": ("Diabetes Prediction", "சர்க்கரை நோய் கணிப்பு"),
    "assistant": ("Health Assistant", "சுகாதார உதவியாளர்"),
    "bmi": ("BMI Calculator", "BMI கணிப்பான்"),
    "history": ("Prediction History", "கணிப்பு வரலாறு"),
    "learning": ("Health Learning Center", "சுகாதாரக் கற்றல் மையம்"),
    "about": ("About", "எங்களைப் பற்றி"),
    "welcome": (
        "Welcome to your intelligent health dashboard.",
        "உங்கள் AI சுகாதார dashboard-க்கு வரவேற்கிறோம்."
    ),
    "overview": ("Your Project Overview", "உங்கள் திட்டத்தின் விவரங்கள்"),
    "model": ("AI Model", "AI மாடல்"),
    "dataset": ("Dataset", "தரவுத்தொகுப்பு"),
    "technology": ("Technology", "தொழில்நுட்பம்"),
    "features": ("Platform Features", "தளத்தின் அம்சங்கள்"),
    "enter_data": (
        "Enter your health measurements below.",
        "கீழே உங்கள் உடல்நல அளவீடுகளை உள்ளிடுங்கள்."
    ),
    "pregnancies": ("Pregnancies", "கர்ப்ப எண்ணிக்கை"),
    "glucose": ("Glucose", "குளுக்கோஸ்"),
    "bp": ("Blood Pressure", "இரத்த அழுத்தம்"),
    "skin": ("Skin Thickness", "தோல் தடிமன்"),
    "insulin": ("Insulin", "இன்சுலின்"),
    "bmi_label": ("BMI", "உடல் நிறை குறியீடு (BMI)"),
    "dpf": ("Diabetes Pedigree Function", "குடும்ப நீரிழிவு தொடர்பான அளவு"),
    "age": ("Age", "வயது"),
    "analyze": ("Analyze Health Data", "உடல்நலத் தரவை ஆய்வு செய்"),
    "result": ("AI Prediction Result", "AI கணிப்பு முடிவு"),
    "higher": (
        "Higher predicted risk category",
        "அதிக கணிப்பு அபாய வகை"
    ),
    "lower": (
        "Lower predicted risk category",
        "குறைந்த கணிப்பு அபாய வகை"
    ),
    "score": ("Model Output Score", "மாடல் வெளியீட்டு மதிப்பு"),
    "history_empty": (
        "No predictions yet. Make your first prediction.",
        "இதுவரை கணிப்புகள் இல்லை. முதல் கணிப்பைச் செய்யுங்கள்."
    ),
    "total": ("Total Predictions", "மொத்த கணிப்புகள்"),
    "download": ("Download CSV Report", "CSV அறிக்கையைப் பதிவிறக்கு"),
    "ask": (
        "Ask about diabetes, BMI, food or exercise...",
        "நீரிழிவு, BMI, உணவு அல்லது உடற்பயிற்சி பற்றி கேளுங்கள்..."
    ),
    "clear": ("Clear Conversation", "உரையாடலை அழி"),
    "bmi_weight": ("Weight (kg)", "எடை (கிலோ)"),
    "bmi_height": ("Height (cm)", "உயரம் (செ.மீ.)"),
    "calculate": ("Calculate BMI", "BMI கணக்கிடு"),
    "your_bmi": ("Your BMI", "உங்கள் BMI"),
    "learning_title": ("Learn About Health", "உடல்நலம் பற்றி அறிந்துகொள்ளுங்கள்"),
    "disclaimer": (
        "Educational demonstration only. This application is not a medical diagnostic tool. Its output is not a clinically validated probability. Consult a qualified healthcare professional for testing and medical advice.",
        "இது கல்வி நோக்கத்திற்கான செயலி மட்டுமே. மருத்துவ நோயறிதல் கருவி அல்ல. இதன் முடிவு மருத்துவ ரீதியாக உறுதிப்படுத்தப்பட்ட தனிப்பட்ட அபாய மதிப்பு அல்ல. பரிசோதனை மற்றும் மருத்துவ ஆலோசனைக்கு தகுதியான மருத்துவரை அணுகவும்."
    ),
    "about_text": (
        "HealthPredict AI is an educational AI/ML project demonstrating diabetes prediction using an Artificial Neural Network.",
        "HealthPredict AI என்பது Artificial Neural Network மூலம் நீரிழிவு கணிப்பை விளக்கும் AI/ML கல்வித் திட்டமாகும்."
    )
}


def T(key):
    language = st.session_state.get("language", "English")
    index = 1 if language == "தமிழ்" else 0
    return TEXT[key][index]


# ---------------- SESSION STATE ----------------

if "language" not in st.session_state:
    st.session_state.language = "English"

if "theme" not in st.session_state:
    st.session_state.theme = "Neon Teal + Purple"

if "history" not in st.session_state:
    st.session_state.history = []

if "chat" not in st.session_state:
    st.session_state.chat = []


# ---------------- SIDEBAR SETTINGS ----------------

st.sidebar.markdown("# 🩺 HealthPredict AI")
st.sidebar.caption("Version 3.0 | AI Healthcare Platform")

st.sidebar.divider()

language_choice = st.sidebar.selectbox(
    T("language"),
    ["English", "தமிழ்"],
    index=0 if st.session_state.language == "English" else 1
)

st.session_state.language = language_choice

theme_choice = st.sidebar.selectbox(
    T("theme"),
    [
        "Neon Teal + Purple",
        "Royal Purple + Black",
        "Clean White + Blue"
    ],
    index=[
        "Neon Teal + Purple",
        "Royal Purple + Black",
        "Clean White + Blue"
    ].index(st.session_state.theme)
)

st.session_state.theme = theme_choice


# ---------------- THEME COLORS ----------------

themes = {
    "Neon Teal + Purple": {
        "bg": "#080d1b",
        "panel": "#142239",
        "accent": "#45e6e0",
        "secondary": "#9b8cff",
        "text": "#f4f7ff"
    },
    "Royal Purple + Black": {
        "bg": "#100d1c",
        "panel": "#211735",
        "accent": "#b18aff",
        "secondary": "#ed8cff",
        "text": "#faf5ff"
    },
    "Clean White + Blue": {
        "bg": "#f3f7ff",
        "panel": "#ffffff",
        "accent": "#2563eb",
        "secondary": "#0ea5e9",
        "text": "#172554"
    }
}

colors = themes[theme_choice]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --background: THEME_BG;
        --panel: THEME_PANEL;
        --accent: THEME_ACCENT;
        --secondary: THEME_SECONDARY;
        --main-text: THEME_TEXT;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, var(--panel), transparent 40%),
            var(--background);
        color: var(--main-text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--accent);
    }

    h1 {
        color: var(--accent) !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: var(--main-text) !important;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--accent);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    }

    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-weight: 800;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(100deg, var(--accent), var(--secondary));
        color: #101426;
        border: none;
        border-radius: 12px;
        min-height: 44px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 18px var(--accent);
    }

    [data-testid="stChatMessage"] {
        background: var(--panel);
        border: 1px solid var(--accent);
        border-radius: 15px;
    }

    [data-testid="stProgressBar"] > div > div {
        background: var(--accent);
    }

    hr {
        border-color: var(--accent);
        opacity: 0.3;
    }

    </style>
    """.replace("THEME_BG", colors["bg"])
       .replace("THEME_PANEL", colors["panel"])
       .replace("THEME_ACCENT", colors["accent"])
       .replace("THEME_SECONDARY", colors["secondary"])
       .replace("THEME_TEXT", colors["text"]),
    unsafe_allow_html=True
)


# ---------------- LOAD EXISTING MODEL ----------------

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("health_predict_model.keras")
    scaler = joblib.load("health_scaler.pkl")
    features = joblib.load("health_features.pkl")

    feature_names = list(features)

    expected_features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    if feature_names != expected_features:
        raise ValueError(
            "Feature order does not match the trained model. "
            "Check health_features.pkl."
        )

    return model, scaler, feature_names


try:
    model, scaler, features = load_resources()
    model_loaded = True
except Exception as error:
    model_loaded = False
    st.error("Unable to load the existing ANN model.")
    st.code(str(error))
    st.info(
        "Check that health_predict_model.keras, "
        "health_scaler.pkl and health_features.pkl "
        "are present in the repository root."
    )
    st.stop()


# ---------------- NAVIGATION ----------------

st.sidebar.divider()

page_options = [
    T("dashboard"),
    T("prediction"),
    T("assistant"),
    T("bmi"),
    T("history"),
    T("learning"),
    T("about")
]

page = st.sidebar.radio(
    T("navigation"),
    page_options
)

st.sidebar.divider()
st.sidebar.caption("Powered by TensorFlow + Streamlit")


# ---------------- DASHBOARD ----------------

if page == T("dashboard"):

    st.title("🩺 HealthPredict AI 3.0")
    st.subheader(T("tagline"))
    st.write(T("welcome"))

    col1, col2, col3 = st.columns(3)

    col1.metric(T("model"), "ANN")
    col2.metric(T("dataset"), "Pima Indians")
    col3.metric(T("technology"), "AI / ML")

    st.divider()
    st.subheader(T("overview"))

    total = len(st.session_state.history)

    high_count = sum(
        1 for item in st.session_state.history
        if item["Result"] == "Higher predicted risk"
    )

    low_count = total - high_count

    a, b, c = st.columns(3)

    a.metric(T("total"), total)
    b.metric("Higher category", high_count)
    c.metric("Lower category", low_count)

    st.divider()
    st.subheader(T("features"))

    card1, card2 = st.columns(2)

    with card1:
        st.info(
            "🔍 **Diabetes Prediction**\n\n"
            "Explore an educational prediction using your trained ANN model."
        )
        st.info(
            "💬 **Health Assistant**\n\n"
            "Learn general health information in Tamil or English."
        )

    with card2:
        st.info(
            "📊 **Analytics**\n\n"
            "Review session prediction history and download CSV."
        )
        st.info(
            "🧮 **BMI Calculator**\n\n"
            "Calculate BMI using height and weight."
        )

    st.warning(T("disclaimer"))


# ---------------- DIABETES PREDICTION ----------------

elif page == T("prediction"):

    st.title("🔍 " + T("prediction"))
    st.write(T("enter_data"))

    with st.form("health_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                T("pregnancies"), 0, 20, 1
            )
            glucose = st.number_input(
                T("glucose"), 0, 300, 120
            )
            blood_pressure = st.number_input(
                T("bp"), 0, 200, 70
            )
            skin_thickness = st.number_input(
                T("skin"), 0, 100, 20
            )

        with col2:
            insulin = st.number_input(
                T("insulin"), 0, 900, 79
            )
            bmi = st.number_input(
                T("bmi_label"), 0.0, 100.0, 25.0
            )
            dpf = st.number_input(
                T("dpf"), 0.0, 3.0, 0.5
            )
            age = st.number_input(
                T("age"), 1, 120, 30
            )

        submitted = st.form_submit_button(
            T("analyze"),
            use_container_width=True
        )

    if submitted:

        input_values = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        input_data = pd.DataFrame(
            [[input_values[name] for name in features]],
            columns=features
        )

        try:
            scaled_data = scaler.transform(input_data)

            probability = float(
                model.predict(scaled_data, verbose=0)[0][0]
            )

            prediction = probability >= 0.5

            st.divider()
            st.subheader(T("result"))

            if prediction:
                st.error(T("higher"))
                result_text = "Higher predicted risk"
            else:
                st.success(T("lower"))
                result_text = "Lower predicted risk"

            st.metric(
                T("score"),
                f"{probability * 100:.2f}%"
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.caption(
                "This is the model's output score, not a clinically "
                "validated personal probability."
            )

            st.session_state.history.append({
                "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Result": result_text,
                "Score": round(probability * 100, 2)
            })

            st.warning(T("disclaimer"))

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))


# ---------------- HEALTH ASSISTANT ----------------

elif page == T("assistant"):

    st.title("💬 " + T("assistant"))

    st.caption(
        "Educational rule-based assistant | "
        "கல்வி நோக்கத்திற்கான உதவியாளர்"
    )

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(T("ask"))

    if question:

        st.session_state.chat.append({
            "role": "user",
            "content": question
        })

        q = question.lower()
        tamil = st.session_state.language == "தமிழ்"

        if any(word in q for word in [
            "symptom", "symptoms", "அறிகுறி", "அறிகுறிகள்"
        ]):
            answer = (
                "நீரிழிவு நோயின் பொதுவான அறிகுறிகளில் அதிக தாகம், "
                "அடிக்கடி சிறுநீர் கழித்தல், சோர்வு மற்றும் மங்கலான பார்வை "
                "ஆகியவை அடங்கும். மருத்துவரிடம் பரிசோதனை பெறுங்கள்."
                if tamil else
                "Common symptoms can include increased thirst, frequent "
                "urination, tiredness and blurred vision. Consult a healthcare "
                "professional for appropriate testing."
            )

        elif any(word in q for word in [
            "food", "diet", "eat", "உணவு", "சாப்பாடு"
        ]):
            answer = (
                "காய்கறிகள், முழு தானியங்கள், பருப்பு வகைகள் மற்றும் "
                "சரியான புரத உணவுகளை உள்ளடக்கிய சமநிலையான உணவு உதவியாக "
                "இருக்கலாம். தனிப்பட்ட உணவுத் திட்டத்திற்கு உணவியல் நிபுணரை அணுகவும்."
                if tamil else
                "A balanced eating pattern may include vegetables, whole "
                "grains, pulses and suitable protein sources. Consult a "
                "registered dietitian for personal dietary advice."
            )

        elif any(word in q for word in [
            "exercise", "walking", "workout", "நடை", "உடற்பயிற்சி"
        ]):
            answer = (
                "நடைப்பயிற்சி, சைக்கிள் ஓட்டுதல் போன்ற உடல் செயல்பாடுகள் "
                "பொது உடல்நலத்திற்கு உதவலாம். உங்கள் உடல்நிலைக்கு ஏற்ற "
                "பயிற்சியைத் தேர்ந்தெடுக்கவும்."
                if tamil else
                "Walking, cycling and other suitable physical activities "
                "can support general health. Choose activity appropriate "
                "for your health and fitness level."
            )

        elif any(word in q for word in [
            "insulin", "இன்சுலின்"
        ]):
            answer = (
                "இன்சுலின் என்பது இரத்தத்தில் உள்ள குளுக்கோஸ் செல்களுக்குள் "
                "செல்ல உதவும் ஒரு ஹார்மோன் ஆகும்."
                if tamil else
                "Insulin is a hormone that helps glucose move from the blood "
                "into body cells."
            )

        elif any(word in q for word in [
            "bmi", "உடல் நிறை"
        ]):
            answer = (
                "BMI என்பது உடல் எடை மற்றும் உயரத்தைப் பயன்படுத்தி "
                "கணக்கிடப்படும் ஒரு அளவீடு. இது முழுமையான உடல்நல மதிப்பீடு அல்ல."
                if tamil else
                "BMI stands for Body Mass Index. It uses weight and height "
                "as a screening measure, not a complete health assessment."
            )

        elif any(word in q for word in [
            "hello", "hi", "hey", "வணக்கம்"
        ]):
            answer = (
                "வணக்கம்! நீரிழிவு, உணவு, உடற்பயிற்சி, BMI பற்றி கேளுங்கள்."
                if tamil else
                "Hello! Ask me about diabetes, food, exercise, BMI or insulin."
            )

        else:
            answer = (
                "நீரிழிவு, அறிகுறிகள், உணவு, உடற்பயிற்சி, BMI மற்றும் "
                "இன்சுலின் பற்றி பொதுவான தகவல்களை வழங்க முடியும். "
                "தனிப்பட்ட மருத்துவ ஆலோசனைக்கு மருத்துவரை அணுகவும்."
                if tamil else
                "I can provide general educational information about diabetes, "
                "symptoms, food, exercise, BMI and insulin. "
                "Consult a healthcare professional for personal medical advice."
            )

        st.session_state.chat.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button(T("clear")):
        st.session_state.chat = []
        st.rerun()


# ---------------- BMI CALCULATOR ----------------

elif page == T("bmi"):

    st.title("🧮 " + T("bmi"))

    weight = st.number_input(
        T("bmi_weight"),
        min_value=1.0,
        max_value=300.0,
        value=65.0
    )

    height = st.number_input(
        T("bmi_height"),
        min_value=50.0,
        max_value=250.0,
        value=170.0
    )

    if st.button(T("calculate")):

        height_m = height / 100
        bmi_value = weight / (height_m ** 2)

        st.metric(
            T("your_bmi"),
            f"{bmi_value:.1f}"
        )

        if bmi_value < 18.5:
            category = "Underweight / குறைந்த எடை"
        elif bmi_value < 25:
            category = "Normal range / இயல்பான வரம்பு"
        elif bmi_value < 30:
            category = "Overweight / அதிக எடை"
        else:
            category = "Obesity range / உடல் பருமன் வரம்பு"

        st.info(category)

        st.caption(
            "BMI is a screening measure, not a diagnosis. "
            "Interpretation may vary by age and individual circumstances."
        )


# ---------------- PREDICTION HISTORY ----------------

elif page == T("history"):

    st.title("📊 " + T("history"))

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.metric(T("total"), len(history_df))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Prediction Score Analytics")

        chart_data = history_df[["Score"]].copy()
        chart_data.index = range(1, len(chart_data) + 1)

        st.line_chart(chart_data)

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            T("download"),
            data=csv_data,
            file_name="healthpredict_history.csv",
            mime="text/csv"
        )

    else:
        st.info(T("history_empty"))


# ---------------- HEALTH LEARNING CENTER ----------------

elif page == T("learning"):

    st.title("📚 " + T("learning_title"))

    if st.session_state.language == "தமிழ்":

        with st.expander("🩺 நீரிழிவு என்றால் என்ன?"):
            st.write(
                "இரத்தத்தில் குளுக்கோஸ் அளவு அதிகமாக இருக்கும் நிலையே "
                "நீரிழிவு நோய். இதைப் பரிசோதனைகள் மூலம் மருத்துவர் மதிப்பிடுவார்."
            )

        with st.expander("🥗 ஆரோக்கியமான உணவு"):
            st.write(
                "காய்கறிகள், முழு தானியங்கள், பருப்பு வகைகள் மற்றும் "
                "சமநிலையான உணவுப் பழக்கங்கள் பொது உடல்நலத்திற்கு உதவும்."
            )

        with st.expander("🚶 உடற்பயிற்சியின் முக்கியத்துவம்"):
            st.write(
                "தொடர்ச்சியான உடல் செயல்பாடு இதயம் மற்றும் ஒட்டுமொத்த "
                "உடல்நலத்திற்கு உதவக்கூடும்."
            )

        with st.expander("🧪 உடல்நலப் பரிசோதனைகள்"):
            st.write(
                "இரத்த குளுக்கோஸ் மற்றும் HbA1c போன்ற பரிசோதனைகள் "
                "நீரிழிவை மதிப்பிட உதவும். மருத்துவரின் ஆலோசனையைப் பெறுங்கள்."
            )

    else:

        with st.expander("🩺 What is Diabetes?"):
            st.write(
                "Diabetes is a condition involving elevated blood glucose. "
                "Healthcare professionals use appropriate tests to assess it."
            )

        with st.expander("🥗 Healthy Eating"):
            st.write(
                "Vegetables, whole grains, pulses and balanced food habits "
                "can support general health."
            )

        with st.expander("🚶 Importance of Exercise"):
            st.write(
                "Regular physical activity can support cardiovascular "
                "health and overall wellbeing."
            )

        with st.expander("🧪 Health Tests"):
            st.write(
                "Blood glucose and HbA1c tests can help healthcare "
                "professionals assess diabetes."
            )

    st.warning(T("disclaimer"))


# ---------------- ABOUT ----------------

elif page == T("about"):

    st.title("ℹ️ " + T("about"))

    st.write(T("about_text"))

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - TensorFlow / Keras
    - Artificial Neural Network
    - Scikit-learn
    - Pandas and NumPy
    - Streamlit
    """)

    st.subheader("Dataset")
    st.write("Pima Indians Diabetes Dataset")

    st.subheader("Developer")
    st.write("Harish S | AI & Data Science")

    st.warning(T("disclaimer"))

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "HealthPredict AI 3.0 | Educational AI/ML Project | "
    "Not a medical diagnostic application"
)
