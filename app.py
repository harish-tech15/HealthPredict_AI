import os
import io
import pickle
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from google import genai
from google.genai import types

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HealthPredict AI 4.0",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "health_predict_model.keras"
SCALER_PATH = "health_scaler.pkl"
FEATURES_PATH = "health_features.pkl"

MODEL_NAME = "gemini-3.8-flash"

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin (mu U/ml)",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age"
}

TEXT = {
    "English": {
        "dashboard": "🏠 Dashboard",
        "prediction": "🩺 Diabetes Risk Prediction",
        "assistant": "🤖 AI Health Assistant",
        "food": "🥗 Healthy Food Guide",
        "exercise": "🏃 Exercise & Fitness",
        "learning": "📚 Health Learning Center",
        "reports": "📊 Reports & Downloads",
        "about": "ℹ️ About",
        "welcome": "Your AI-powered health and wellness companion",
        "language": "Language",
        "theme": "Theme",
        "ask": "Ask a health-related question...",
        "send": "Send message",
        "clear": "Clear chat",
        "bmi": "BMI Calculator",
        "calculate": "Calculate BMI",
        "name": "Name (optional)",
        "age": "Age",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "predict": "Predict diabetes risk",
        "result": "Prediction result",
        "low": "Lower predicted risk",
        "high": "Higher predicted risk",
        "report": "Download your health report",
        "disclaimer": "Educational information only. Not a medical diagnosis."
    },
    "தமிழ்": {
        "dashboard": "🏠 முகப்பு",
        "prediction": "🩺 நீரிழிவு அபாய கணிப்பு",
        "assistant": "🤖 AI சுகாதார உதவியாளர்",
        "food": "🥗 ஆரோக்கிய உணவு வழிகாட்டி",
        "exercise": "🏃 உடற்பயிற்சி & உடல்நலம்",
        "learning": "📚 சுகாதார அறிவு மையம்",
        "reports": "📊 அறிக்கைகள் & பதிவிறக்கம்",
        "about": "ℹ️ எங்களைப் பற்றி",
        "welcome": "உங்கள் AI சுகாதார மற்றும் நலவாழ்வு உதவியாளர்",
        "language": "மொழி",
        "theme": "நிறத் தீம்",
        "ask": "உடல்நலம் தொடர்பான கேள்வியைக் கேளுங்கள்...",
        "send": "செய்தி அனுப்பு",
        "clear": "உரையாடலை அழி",
        "bmi": "BMI கணக்கீடு",
        "calculate": "BMI கணக்கிடு",
        "name": "பெயர் (விருப்பம்)",
        "age": "வயது",
        "height": "உயரம் (செ.மீ.)",
        "weight": "எடை (கிலோ)",
        "predict": "நீரிழிவு அபாயத்தைக் கணிக்கவும்",
        "result": "கணிப்பு முடிவு",
        "low": "குறைவான கணிக்கப்பட்ட அபாயம்",
        "high": "அதிகமான கணிக்கப்பட்ட அபாயம்",
        "report": "சுகாதார அறிக்கையைப் பதிவிறக்கவும்",
        "disclaimer": "இது கல்வி நோக்கத்திற்கான தகவல் மட்டுமே. மருத்துவ நோயறிதல் அல்ல."
    }
}


def T(key):
    return TEXT[language].get(key, key)


# --------------------------------------------------
# 2. SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.markdown("# 🩺 HealthPredict AI")
st.sidebar.caption("Version 4.0 | Smart Health & Wellness")

language = st.sidebar.selectbox(
    "🌐 Language / மொழி",
    ["English", "தமிழ்"],
    index=0
)

theme = st.sidebar.selectbox(
    "🎨 Theme",
    [
        "Neon Teal + Purple",
        "Royal Purple + Black",
        "Ocean Blue",
        "Clean Light"
    ]
)

themes = {
    "Neon Teal + Purple": {
        "bg": "#07131F",
        "card": "#102536",
        "primary": "#22D3C5",
        "secondary": "#A78BFA",
        "text": "#F1F5F9"
    },
    "Royal Purple + Black": {
        "bg": "#100B1D",
        "card": "#211633",
        "primary": "#C084FC",
        "secondary": "#F0ABFC",
        "text": "#FAF5FF"
    },
    "Ocean Blue": {
        "bg": "#071B32",
        "card": "#102D4C",
        "primary": "#38BDF8",
        "secondary": "#67E8F9",
        "text": "#F0F9FF"
    },
    "Clean Light": {
        "bg": "#F3F7FB",
        "card": "#FFFFFF",
        "primary": "#087E8B",
        "secondary": "#6655CC",
        "text": "#172033"
    }
}

colorset = themes[theme]

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {colorset['bg']};
        color: {colorset['text']};
    }}
    [data-testid="stSidebar"] {{
        background: {colorset['card']};
    }}
    .hero {{
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(
            120deg,
            {colorset['card']},
            {colorset['bg']}
        );
        border: 1px solid {colorset['primary']};
        margin-bottom: 20px;
    }}
    .hero h1 {{
        color: {colorset['primary']};
        margin-bottom: 5px;
    }}
    .feature-card {{
        padding: 20px;
        border-radius: 16px;
        background: {colorset['card']};
        border: 1px solid {colorset['secondary']};
        min-height: 130px;
        margin-bottom: 12px;
    }}
    .feature-card h3 {{
        color: {colorset['primary']};
    }}
    div.stButton > button {{
        border-radius: 12px;
        border: 1px solid {colorset['primary']};
        font-weight: 600;
    }}
    div[data-testid="stMetric"] {{
        background: {colorset['card']};
        padding: 15px;
        border-radius: 14px;
        border: 1px solid {colorset['primary']};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 3. LOAD EXISTING DIABETES MODEL FILES
# --------------------------------------------------

@st.cache_resource
def load_diabetes_files():
    required = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]

    missing = [file for file in required if not os.path.exists(file)]

    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(FEATURES_PATH, "rb") as file:
        feature_names = pickle.load(file)

    return model, scaler, list(feature_names)


model = None
scaler = None
feature_names = None
model_error = None

try:
    model, scaler, feature_names = load_diabetes_files()
except Exception as error:
    model_error = str(error)


# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# --------------------------------------------------
# 5. NAVIGATION
# --------------------------------------------------

pages = [
    T("dashboard"),
    T("prediction"),
    T("assistant"),
    T("food"),
    T("exercise"),
    T("learning"),
    T("reports"),
    T("about")
]

page = st.sidebar.radio("Navigation", pages)

st.sidebar.divider()
st.sidebar.caption("🔐 Your API key is stored in Streamlit Secrets.")
st.sidebar.caption("⚠️ For educational use only.")


# --------------------------------------------------
# 6. HELPER FUNCTIONS
# --------------------------------------------------

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m * height_m)


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal range"
    elif bmi < 30:
        return "Overweight"
    return "Obesity range"


def get_gemini_client():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def ask_gemini(question):
    client = get_gemini_client()

    if client is None:
        return (
            "Gemini API key is missing. Add GOOGLE_API_KEY "
            "in Streamlit Cloud → App settings → Secrets."
        )

    system_instruction = """
    You are HealthPredict AI, a careful health education assistant.

    Rules:
    1. Answer only health, wellness, nutrition, fitness and medical
       education questions.
    2. If asked about unrelated topics, politely explain that you
       are designed for health-related questions.
    3. Support Tamil and English. Reply in the language used by
       the user.
    4. Explain common symptoms and possible causes as general
       information, never as a confirmed diagnosis.
    5. Never claim to detect or diagnose every disease.
    6. Never prescribe medicines, dosages or treatment changes.
    7. Encourage users to consult a qualified healthcare professional.
    8. For emergency warning signs such as severe chest pain,
       difficulty breathing, stroke symptoms, unconsciousness or
       severe bleeding, advise immediate emergency medical help.
    9. Give practical, safe, evidence-informed general guidance.
    10. Keep answers clear and easy to understand.
    """

    previous_messages = ""

    for item in st.session_state.chat_history[-8:]:
        previous_messages += (
            f"\n{item['role'].upper()}: {item['content']}\n"
        )

    prompt = (
        previous_messages
        + "\nUSER: "
        + question
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
            max_output_tokens=1200
        )
    )

    return response.text or "Sorry, I could not generate a response."


def create_pdf_report():
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#087E8B")
    )

    story = [
        Paragraph("HealthPredict AI - Health Report", title_style),
        Spacer(1, 12),
        Paragraph(
            "Generated: " + datetime.now().strftime("%d-%m-%Y %H:%M"),
            styles["Normal"]
        ),
        Spacer(1, 15)
    ]

    last = st.session_state.last_prediction

    if last:
        report_data = [
            ["Item", "Value"],
            ["Diabetes risk result", last["result"]],
            ["Model score", last["score"]],
            ["Age", last["age"]],
            ["BMI", last["bmi"]]
        ]

        table = Table(report_data, colWidths=[220, 220])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087E8B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ]))

        story.append(Paragraph("Latest Prediction", styles["Heading2"]))
        story.append(table)
    else:
        story.append(
            Paragraph(
                "No diabetes prediction has been recorded in this session.",
                styles["Normal"]
            )
        )

    story.append(Spacer(1, 18))
    story.append(Paragraph("Prediction History", styles["Heading2"]))

    history_data = [["Date", "Result", "Score"]]

    for item in st.session_state.history:
        history_data.append([
            item["date"],
            item["result"],
            item["score"]
        ])

    history_table = Table(history_data, repeatRows=1)
    history_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6655CC")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(history_table)
    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "IMPORTANT: This report is for educational purposes only. "
            "It is not a medical diagnosis. Consult a qualified healthcare "
            "professional for medical advice.",
            styles["Italic"]
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def create_excel_report():
    workbook = Workbook()

    summary = workbook.active
    summary.title = "Health Summary"

    summary.append(["HealthPredict AI - Health Summary"])
    summary.append(["Generated", datetime.now().strftime("%d-%m-%Y %H:%M")])

    last = st.session_state.last_prediction

    if last:
        summary.append(["Latest prediction", last["result"]])
        summary.append(["Model score", last["score"]])
        summary.append(["Age", last["age"]])
        summary.append(["BMI", last["bmi"]])
    else:
        summary.append(["Latest prediction", "No prediction yet"])

    history_sheet = workbook.create_sheet("Prediction History")
    history_sheet.append(["Date", "Result", "Score", "Age", "BMI"])

    for item in st.session_state.history:
        history_sheet.append([
            item["date"],
            item["result"],
            item["score"],
            item["age"],
            item["bmi"]
        ])

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(
                start_color="087E8B",
                end_color="087E8B",
                fill_type="solid"
            )
            cell.alignment = Alignment(horizontal="center")

        for column in sheet.columns:
            max_length = max(
                len(str(cell.value or "")) for cell in column
            )
            sheet.column_dimensions[column[0].column_letter].width = (
                min(max_length + 4, 35)
            )

    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return output.getvalue()


# --------------------------------------------------
# 7. DASHBOARD
# --------------------------------------------------

if page == T("dashboard"):

    st.markdown(
        f"""
        <div class="hero">
            <h1>🩺 HealthPredict AI 4.0</h1>
            <h3>{T("welcome")}</h3>
            <p>🤖 AI Assistant | 🥗 Nutrition | 🏃 Fitness | 📊 Reports</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("🩺 Predictions", len(st.session_state.history))
    col2.metric("💬 Chat messages", len(st.session_state.chat_history))
    col3.metric(
        "🧠 Diabetes model",
        "Loaded" if model is not None else "Not loaded"
    )

    st.subheader("✨ Explore HealthPredict AI")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            '<div class="feature-card"><h3>🤖 AI Health Assistant</h3>'
            '<p>Ask health-related questions in Tamil or English.</p></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="feature-card"><h3>🥗 Healthy Food Guide</h3>'
            '<p>Explore balanced meals and nutrition tips.</p></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="feature-card"><h3>🩺 Diabetes Risk Prediction</h3>'
            '<p>Use your existing trained ANN model.</p></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="feature-card"><h3>📊 Health Reports</h3>'
            '<p>Download PDF and Excel summaries.</p></div>',
            unsafe_allow_html=True
        )

    if model_error:
        st.warning("Diabetes model could not be loaded: " + model_error)

    st.info(T("disclaimer"))


# --------------------------------------------------
# 8. DIABETES RISK PREDICTION
# --------------------------------------------------

elif page == T("prediction"):

    st.title(T("prediction"))
    st.info(
        "This prediction uses your existing diabetes model. "
        "It does not detect other diseases."
    )

    if model is None:
        st.error(
            "Model files could not be loaded. Check that all three "
            "model files are in the same GitHub folder as app.py."
        )
        st.code(model_error or "Unknown model loading error")

    else:
        with st.form("diabetes_form"):

            st.subheader("🧍 Enter Health Details")

            c1, c2 = st.columns(2)

            with c1:
                pregnancies = st.number_input(
                    "Pregnancies",
                    min_value=0,
                    max_value=20,
                    value=0
                )
                glucose = st.number_input(
                    "Glucose (mg/dL)",
                    min_value=0.0,
                    max_value=500.0,
                    value=120.0
                )
                bp = st.number_input(
                    "Blood Pressure (mm Hg)",
                    min_value=0.0,
                    max_value=250.0,
                    value=70.0
                )
                skin = st.number_input(
                    "Skin Thickness (mm)",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0
                )

            with c2:
                insulin = st.number_input(
                    "Insulin (mu U/ml)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=80.0
                )
                bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0
                )
                dpf = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.5
                )
                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=25
                )

            submitted = st.form_submit_button(
                "🔍 " + T("predict"),
                use_container_width=True
            )

        if submitted:
            input_values = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age
            }

            try:
                if set(feature_names) != set(input_values.keys()):
                    raise ValueError(
                        "The saved feature names do not match the "
                        "expected diabetes model inputs."
                    )

                input_df = pd.DataFrame(
                    [[input_values[name] for name in feature_names]],
                    columns=feature_names
                )

                scaled_input = scaler.transform(input_df)
                probability = float(
                    model.predict(scaled_input, verbose=0)[0][0]
                )

                result = (
                    "Higher predicted risk"
                    if probability >= 0.5
                    else "Lower predicted risk"
                )

                score = round(probability * 100, 2)

                st.session_state.last_prediction = {
                    "result": result,
                    "score": score,
                    "age": age,
                    "bmi": bmi
                }

                record = {
                    "date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "result": result,
                    "score": score,
                    "age": age,
                    "bmi": bmi
                }

                st.session_state.history.append(record)

                st.subheader("📊 " + T("result"))

                if probability >= 0.5:
                    st.warning("⚠️ " + T("high"))
                else:
                    st.success("✅ " + T("low"))

                st.metric("Model score", f"{score}%")
                st.progress(min(max(probability, 0.0), 1.0))

                st.caption(
                    "This is the model's estimated probability, not "
                    "your confirmed chance of having diabetes."
                )

                st.info(
                    "Please discuss your result with a qualified "
                    "healthcare professional. A blood test and clinical "
                    "assessment are needed for diagnosis."
                )

            except Exception as error:
                st.error("Prediction error: " + str(error))


# --------------------------------------------------
# 9. GEMINI AI HEALTH ASSISTANT
# --------------------------------------------------

elif page == T("assistant"):

    st.title(T("assistant"))
    st.caption("Powered by Google Gemini | Tamil + English")

    st.info(
        "Ask about general health, diseases, symptoms, nutrition, "
        "fitness and prevention. This assistant cannot diagnose diseases."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(T("ask"))

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = ask_gemini(question)
                except Exception as error:
                    answer = (
                        "Gemini request failed. Please check your API key "
                        "and try again.\n\nError: " + str(error)
                    )

                st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

    if st.button("🗑️ " + T("clear")):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------
# 10. HEALTHY FOOD GUIDE
# --------------------------------------------------

elif page == T("food"):

    st.title(T("food"))
    st.write("🥑 Explore balanced food choices and everyday nutrition.")

    st.subheader("🍽️ Balanced Indian Meal Ideas")

    meals = {
        "🌅 Breakfast": [
            "Idli with sambar",
            "Vegetable dosa",
            "Oats with nuts",
            "Boiled eggs with whole-grain toast"
        ],
        "☀️ Lunch": [
            "Rice in a suitable portion",
            "Vegetable poriyal",
            "Sambar or dal",
            "Curd and salad"
        ],
        "🌙 Dinner": [
            "Chapati with vegetable kurma",
            "Vegetable soup with protein source",
            "Millet dosa with sambar"
        ],
        "🍎 Snacks": [
            "Whole fruit",
            "Roasted chana",
            "Unsalted nuts",
            "Buttermilk without added sugar"
        ]
    }

    for meal, items in meals.items():
        with st.expander(meal, expanded=True):
            for item in items:
                st.write("• " + item)

    st.subheader("💡 Healthy Eating Tips")
    st.markdown("""
    - 🥦 Include different vegetables and fruits.
    - 🫘 Add protein sources such as dal, beans, eggs or fish.
    - 🌾 Prefer whole grains and fibre-rich foods.
    - 💧 Drink water regularly.
    - 🍬 Limit sugary drinks and highly processed snacks.
    - ⚖️ Choose portions that suit your individual needs.
    """)

    st.warning(
        "People with diabetes, kidney disease, food allergies or other "
        "medical conditions may need individual diet advice from a "
        "registered dietitian or doctor."
    )


# --------------------------------------------------
# 11. EXERCISE & FITNESS
# --------------------------------------------------

elif page == T("exercise"):

    st.title(T("exercise"))

    st.subheader("🚶 Beginner Weekly Activity Ideas")

    workout = pd.DataFrame({
        "Day": [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ],
        "Activity": [
            "Brisk walking",
            "Gentle bodyweight exercises",
            "Walking + stretching",
            "Yoga or mobility",
            "Brisk walking",
            "Light strength exercises",
            "Rest or gentle walk"
        ],
        "Suggested duration": [
            "20 minutes", "15 minutes", "20 minutes", "15 minutes",
            "25 minutes", "15 minutes", "As comfortable"
        ]
    })

    st.dataframe(workout, use_container_width=True, hide_index=True)

    st.subheader("🏋️ Simple Beginner Exercises")

    exercises = [
        ("🚶 Walking", "Start at a comfortable pace."),
        ("🪑 Chair Squats", "Use a stable chair for support."),
        ("🧱 Wall Push-ups", "Keep your body straight and move gently."),
        ("🧘 Stretching", "Avoid bouncing or forcing a stretch."),
        ("🧍 Standing March", "March slowly while holding support if needed.")
    ]

    for name, description in exercises:
        st.markdown(f"**{name}** — {description}")

    st.info(
        "Start slowly and increase activity gradually. Stop if you feel "
        "chest pain, faintness or unusual breathlessness. Ask a healthcare "
        "professional before starting exercise if you have a medical "
        "condition or exercise-related symptoms."
    )


# --------------------------------------------------
# 12. HEALTH LEARNING CENTER
# --------------------------------------------------

elif page == T("learning"):

    st.title(T("learning"))

    topics = {
        "🩸 Diabetes": (
            "Diabetes is a condition involving high blood glucose. "
            "There are different types, including type 1, type 2 and "
            "gestational diabetes. Diagnosis requires appropriate "
            "laboratory tests."
        ),
        "❤️ Blood Pressure": (
            "Blood pressure measures the force of blood against artery "
            "walls. Persistently high blood pressure can increase health "
            "risks. Proper measurement and clinical assessment matter."
        ),
        "🫀 Heart Health": (
            "Heart health can be supported through regular physical "
            "activity, balanced food choices, avoiding tobacco and "
            "routine health checks."
        ),
        "🫁 Respiratory Health": (
            "Respiratory conditions affect breathing and the lungs. "
            "Severe difficulty breathing requires urgent medical help."
        ),
        "🧠 Mental Wellbeing": (
            "Sleep, social connection, physical activity and seeking "
            "professional support when needed can support mental wellbeing."
        ),
        "🫘 Kidney Health": (
            "Kidneys help filter waste and regulate fluid balance. "
            "Blood and urine tests may be used by clinicians to assess "
            "kidney function."
        )
    }

    for topic, description in topics.items():
        with st.expander(topic):
            st.write(description)

    st.warning(
        "This learning center provides general education, not diagnosis "
        "or treatment advice."
    )


# --------------------------------------------------
# 13. REPORTS & DOWNLOADS
# --------------------------------------------------

elif page == T("reports"):

    st.title(T("reports"))
    st.write(T("report"))

    if st.session_state.last_prediction:
        last = st.session_state.last_prediction

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest result", last["result"])
        c2.metric("Model score", f"{last['score']}%")
        c3.metric("BMI", last["bmi"])

    else:
        st.info(
            "No prediction yet. You can still download an empty report. "
            "Run a prediction to include your latest result."
        )

    st.subheader("📄 PDF Report")

    pdf_data = create_pdf_report()

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_data,
        file_name="HealthPredict_AI_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.subheader("📊 Excel Report")

    excel_data = create_excel_report()

    st.download_button(
        label="⬇️ Download Excel Report",
        data=excel_data,
        file_name="HealthPredict_AI_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    if st.session_state.history:
        st.subheader("📈 Prediction History")
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# 14. ABOUT
# --------------------------------------------------

elif page == T("about"):

    st.title("ℹ️ About HealthPredict AI 4.0")

    st.markdown("""
    **HealthPredict AI** is an educational health and wellness project.

    ### Main modules
    - 🩺 Diabetes risk estimation using an existing ANN model
    - 🤖 Gemini-powered health education assistant
    - 🥗 Healthy food and nutrition information
    - 🏃 Beginner exercise and fitness guidance
    - 📚 General health learning center
    - 📊 PDF and Excel report downloads
    - 🌐 Tamil and English interface
    - 🎨 Multiple color themes

    ### Technology
    - Python
    - Streamlit
    - TensorFlow / Keras
    - Scikit-learn
    - Google Gemini API
    - Pandas and NumPy
    - ReportLab and OpenPyXL
    """)

    st.error(
        "MEDICAL DISCLAIMER: HealthPredict AI is not a hospital, doctor "
        "or diagnostic system. Its diabetes model estimates risk only. "
        "It does not diagnose diabetes or detect all diseases. Do not "
        "use its outputs to start, stop or change medication. Consult "
        "a qualified healthcare professional for medical decisions. "
        "For emergencies, contact local emergency services."
    )


# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🩺 HealthPredict AI 4.0 | Built for health education and wellness | "
    "Not a substitute for professional medical care."
)
