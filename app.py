
import os
import io
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

# Gemini import is optional so the whole app can still open
# even if the Gemini SDK has not installed correctly.
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
    GEMINI_IMPORT_ERROR = None
except Exception as e:
    genai = None
    types = None
    GEMINI_AVAILABLE = False
    GEMINI_IMPORT_ERROR = str(e)


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HealthPredict AI 4.0",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "health_predict_model.keras"
SCALER_PATH = "health_scaler.pkl"
FEATURES_PATH = "health_features.pkl"

GEMINI_MODEL = "gemini-3.8-flash"

EXPECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


# =========================================================
# 2. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================================================
# 3. SIDEBAR SETTINGS
# =========================================================

st.sidebar.markdown("# 🩺 HealthPredict AI")
st.sidebar.caption("Version 4.0 | Smart Health & Wellness")

language = st.sidebar.selectbox(
    "🌐 Language / மொழி",
    ["English", "தமிழ்"]
)

theme = st.sidebar.selectbox(
    "🎨 Choose Theme",
    [
        "Neon Teal + Purple",
        "Royal Purple + Black",
        "Ocean Blue",
        "Clean Light"
    ]
)

TAMIL = language == "தமிழ்"

TEXT = {
    "dashboard": "🏠 முகப்பு" if TAMIL else "🏠 Dashboard",
    "prediction": "🩺 நீரிழிவு அபாய கணிப்பு" if TAMIL else "🩺 Diabetes Risk Prediction",
    "assistant": "🤖 AI சுகாதார உதவியாளர்" if TAMIL else "🤖 AI Health Assistant",
    "food": "🥗 ஆரோக்கிய உணவு" if TAMIL else "🥗 Healthy Food Guide",
    "exercise": "🏃 உடற்பயிற்சி" if TAMIL else "🏃 Exercise & Fitness",
    "learning": "📚 சுகாதார அறிவு" if TAMIL else "📚 Health Learning Center",
    "reports": "📊 அறிக்கைகள்" if TAMIL else "📊 Reports & Downloads",
    "about": "ℹ️ எங்களைப் பற்றி" if TAMIL else "ℹ️ About",
    "ask": "உடல்நலம் தொடர்பான கேள்வியைக் கேளுங்கள்..." if TAMIL else "Ask a health-related question...",
    "predict": "கணிக்கவும்" if TAMIL else "Predict diabetes risk",
    "clear": "உரையாடலை அழி" if TAMIL else "Clear chat",
    "disclaimer": "கல்வி நோக்கத்திற்கான தகவல் மட்டுமே. மருத்துவ நோயறிதல் அல்ல." if TAMIL else "Educational information only. Not a medical diagnosis."
}

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

C = themes[theme]

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {C['bg']};
        color: {C['text']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {C['card']};
    }}
    .hero {{
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(120deg, {C['card']}, {C['bg']});
        border: 1px solid {C['primary']};
        margin-bottom: 20px;
    }}
    .hero h1 {{
        color: {C['primary']};
    }}
    .feature-card {{
        padding: 20px;
        border-radius: 16px;
        background: {C['card']};
        border: 1px solid {C['secondary']};
        min-height: 130px;
        margin-bottom: 12px;
    }}
    .feature-card h3 {{
        color: {C['primary']};
    }}
    div.stButton > button {{
        border-radius: 12px;
        border: 1px solid {C['primary']};
        font-weight: 600;
    }}
    div[data-testid="stMetric"] {{
        background: {C['card']};
        padding: 14px;
        border-radius: 14px;
        border: 1px solid {C['primary']};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. LOAD EXISTING MODEL FILES
# =========================================================

@st.cache_resource
def load_diabetes_files():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH
    ]

    missing = [
        name for name in required_files
        if not os.path.isfile(name)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files: " + ", ".join(missing)
        )

    # Load the existing trained ANN model.
    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    # Load the existing scaler and feature list.
    # joblib can load objects saved using joblib.dump.
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_features = joblib.load(FEATURES_PATH)

    if isinstance(loaded_features, np.ndarray):
        loaded_features = loaded_features.tolist()

    loaded_features = list(loaded_features)

    if set(loaded_features) != set(EXPECTED_FEATURES):
        raise ValueError(
            "Feature names do not match the expected diabetes model inputs. "
            f"Found: {loaded_features}"
        )

    return loaded_model, loaded_scaler, loaded_features


model = None
scaler = None
feature_names = None
model_error = None

try:
    model, scaler, feature_names = load_diabetes_files()
except Exception as e:
    model_error = str(e)


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    if height_m <= 0:
        return 0
    return weight / (height_m ** 2)


def get_gemini_client():

    if not GEMINI_AVAILABLE:
        raise RuntimeError(
            "Gemini SDK import failed: " + str(GEMINI_IMPORT_ERROR)
        )

    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it in Streamlit Cloud → "
            "Manage app → Settings → Secrets."
        )

    return genai.Client(api_key=api_key)


def ask_gemini(question):

    client = get_gemini_client()

    system_instruction = """
    You are HealthPredict AI, a health education assistant.

    IMPORTANT RULES:
    1. Answer only health, disease education, nutrition, fitness,
       prevention and general wellness questions.
    2. Politely refuse unrelated questions and explain that you are
       a health-focused assistant.
    3. Reply in the same language as the user: Tamil or English.
    4. Explain symptoms as general information, not a diagnosis.
    5. Never claim that you can detect or diagnose every disease.
    6. Never prescribe medicines or dosages.
    7. Never tell users to start, stop or change medication.
    8. Encourage consultation with qualified healthcare professionals.
    9. For chest pain, severe breathing difficulty, stroke symptoms,
       unconsciousness or severe bleeding, advise immediate emergency
       medical help.
    10. Give clear, practical and safe educational information.
    11. Do not claim that the diabetes prediction model detects other
        diseases.
    """

    conversation = ""

    for item in st.session_state.chat_history[-8:]:
        conversation += (
            "\n" + item["role"].upper() + ": " + item["content"]
        )

    prompt = (
        conversation
        + "\nUSER: "
        + question
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4,
            max_output_tokens=1200
        )
    )

    answer = response.text

    if not answer:
        return "Sorry, no response was generated. Please try again."

    return answer


# =========================================================
# 6. PDF REPORT GENERATOR
# =========================================================

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
        data = [
            ["Item", "Value"],
            ["Diabetes risk result", last["result"]],
            ["Model score", str(last["score"]) + "%"],
            ["Age", str(last["age"])],
            ["BMI", str(last["bmi"])]
        ]

        table = Table(data, colWidths=[220, 220])

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
            Paragraph("No prediction recorded yet.", styles["Normal"])
        )

    story.append(Spacer(1, 18))
    story.append(Paragraph("Prediction History", styles["Heading2"]))

    history_data = [["Date", "Result", "Score"]]

    for item in st.session_state.history:
        history_data.append([
            item["date"],
            item["result"],
            str(item["score"]) + "%"
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


# =========================================================
# 7. EXCEL REPORT GENERATOR
# =========================================================

def create_excel_report():

    workbook = Workbook()

    summary = workbook.active
    summary.title = "Health Summary"

    summary.append(["HealthPredict AI - Health Summary"])
    summary.append([
        "Generated",
        datetime.now().strftime("%d-%m-%Y %H:%M")
    ])

    last = st.session_state.last_prediction

    if last:
        summary.append(["Latest prediction", last["result"]])
        summary.append(["Model score", str(last["score"]) + "%"])
        summary.append(["Age", last["age"]])
        summary.append(["BMI", last["bmi"]])
    else:
        summary.append(["Latest prediction", "No prediction yet"])

    history_sheet = workbook.create_sheet("Prediction History")

    history_sheet.append([
        "Date", "Result", "Score", "Age", "BMI"
    ])

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
            sheet.column_dimensions[
                column[0].column_letter
            ].width = min(max_length + 4, 35)

    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)

    return output.getvalue()


# =========================================================
# 8. NAVIGATION
# =========================================================

pages = [
    TEXT["dashboard"],
    TEXT["prediction"],
    TEXT["assistant"],
    TEXT["food"],
    TEXT["exercise"],
    TEXT["learning"],
    TEXT["reports"],
    TEXT["about"]
]

page = st.sidebar.radio("Navigation", pages)

st.sidebar.divider()
st.sidebar.caption("🔐 API key is stored in Streamlit Secrets.")
st.sidebar.caption("⚠️ Educational use only.")


# =========================================================
# 9. DASHBOARD
# =========================================================

if page == TEXT["dashboard"]:

    st.markdown(
        f"""
        <div class="hero">
            <h1>🩺 HealthPredict AI 4.0</h1>
            <h3>Your AI-powered health and wellness companion</h3>
            <p>🤖 AI Assistant | 🥗 Nutrition | 🏃 Fitness | 📊 Reports</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("🩺 Predictions", len(st.session_state.history))
    c2.metric("💬 Chat messages", len(st.session_state.chat_history))
    c3.metric(
        "🧠 Diabetes model",
        "Loaded" if model is not None else "Not loaded"
    )

    st.subheader("✨ Explore HealthPredict AI")

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🤖 AI Health Assistant</h3>
                <p>Ask health questions in Tamil and English.</p>
            </div>
            <div class="feature-card">
                <h3>🥗 Healthy Food Guide</h3>
                <p>Explore balanced meals and nutrition tips.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🩺 Diabetes Risk Prediction</h3>
                <p>Use your existing trained ANN model.</p>
            </div>
            <div class="feature-card">
                <h3>📊 Health Reports</h3>
                <p>Download PDF and Excel summaries.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if model_error:
        st.warning("Diabetes model loading issue: " + model_error)

    st.info(TEXT["disclaimer"])


# =========================================================
# 10. DIABETES RISK PREDICTION
# =========================================================

elif page == TEXT["prediction"]:

    st.title(TEXT["prediction"])

    st.info(
        "This prediction uses your existing diabetes model only. "
        "It does not detect other diseases."
    )

    if model is None:

        st.error(
            "Diabetes model could not be loaded. "
            "Check the three model files in your GitHub repository."
        )

        st.code(model_error or "Unknown model error")

    else:

        with st.form("diabetes_prediction_form"):

            st.subheader("🧍 Enter Health Details")

            left, right = st.columns(2)

            with left:
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

            with right:
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
                "🔍 " + TEXT["predict"],
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
                input_df = pd.DataFrame(
                    [[input_values[name] for name in feature_names]],
                    columns=feature_names
                )

                scaled_input = scaler.transform(input_df)

                probability = float(
                    model.predict(scaled_input, verbose=0).reshape(-1)[0]
                )

                probability = min(max(probability, 0.0), 1.0)

                result = (
                    "Higher predicted risk"
                    if probability >= 0.5
                    else "Lower predicted risk"
                )

                score = round(probability * 100, 2)

                record = {
                    "date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "result": result,
                    "score": score,
                    "age": int(age),
                    "bmi": float(bmi)
                }

                st.session_state.last_prediction = record
                st.session_state.history.append(record)

                st.subheader("📊 Prediction Result")

                if probability >= 0.5:
                    st.warning("⚠️ Higher predicted risk")
                else:
                    st.success("✅ Lower predicted risk")

                st.metric("Model score", f"{score}%")
                st.progress(probability)

                st.caption(
                    "This is the model's estimated probability, not a "
                    "confirmed chance of having diabetes."
                )

                st.info(
                    "A clinical assessment and appropriate blood tests "
                    "are needed for diagnosis. Discuss your result with "
                    "a qualified healthcare professional."
                )

            except Exception as e:
                st.error("Prediction error: " + str(e))


# =========================================================
# 11. GEMINI AI HEALTH ASSISTANT
# =========================================================

elif page == TEXT["assistant"]:

    st.title(TEXT["assistant"])

    st.caption("Powered by Google Gemini | Tamil + English")

    st.info(
        "Ask about health, diseases, symptoms, nutrition, fitness "
        "and prevention. This assistant cannot diagnose diseases."
    )

    if not GEMINI_AVAILABLE:
        st.error(
            "Gemini SDK could not be imported. Check requirements.txt "
            "and Streamlit Cloud logs."
        )
        st.code(str(GEMINI_IMPORT_ERROR))

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(TEXT["ask"])

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("🤖 Gemini is thinking..."):

                try:
                    answer = ask_gemini(question)

                except Exception as e:
                    answer = (
                        "Gemini request failed. Check your API key, "
                        "model access and internet connection.\n\n"
                        + str(e)
                    )

                st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

    if st.button("🗑️ " + TEXT["clear"]):
        st.session_state.chat_history = []
        st.rerun()


# =========================================================
# 12. HEALTHY FOOD GUIDE
# =========================================================

elif page == TEXT["food"]:

    st.title(TEXT["food"])

    st.write(
        "🥑 Simple balanced Indian meal ideas for general wellness."
    )

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
            "Vegetable soup with a protein source",
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
    - 🫘 Include protein sources such as dal, beans, eggs or fish.
    - 🌾 Choose whole grains and fibre-rich foods.
    - 💧 Drink water regularly.
    - 🍬 Limit sugary drinks and highly processed snacks.
    - ⚖️ Choose portions suited to your individual needs.
    """)

    st.warning(
        "People with diabetes, kidney disease, food allergies or other "
        "medical conditions may need individual diet advice from a "
        "qualified healthcare professional or registered dietitian."
    )


# =========================================================
# 13. EXERCISE & FITNESS
# =========================================================

elif page == TEXT["exercise"]:

    st.title(TEXT["exercise"])

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

    st.dataframe(
        workout,
        use_container_width=True,
        hide_index=True
    )

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


# =========================================================
# 14. HEALTH LEARNING CENTER
# =========================================================

elif page == TEXT["learning"]:

    st.title(TEXT["learning"])

    topics = {
        "🩸 Diabetes": (
            "Diabetes is a condition involving high blood glucose. "
            "Types include type 1, type 2 and gestational diabetes. "
            "Diagnosis requires appropriate laboratory tests."
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
        "This learning center provides general education, "
        "not diagnosis or treatment advice."
    )


# =========================================================
# 15. REPORTS & DOWNLOADS
# =========================================================

elif page == TEXT["reports"]:

    st.title(TEXT["reports"])

    last = st.session_state.last_prediction

    if last:

        c1, c2, c3 = st.columns(3)

        c1.metric("Latest result", last["result"])
        c2.metric("Model score", f"{last['score']}%")
        c3.metric("BMI", last["bmi"])

    else:

        st.info(
            "No prediction yet. Run a prediction to include its result "
            "in your report."
        )

    st.subheader("📄 PDF Report")

    try:
        pdf_data = create_pdf_report()

        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_data,
            file_name="HealthPredict_AI_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except Exception as e:
        st.error("PDF report error: " + str(e))

    st.subheader("📊 Excel Report")

    try:
        excel_data = create_excel_report()

        st.download_button(
            "⬇️ Download Excel Report",
            data=excel_data,
            file_name="HealthPredict_AI_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    except Exception as e:
        st.error("Excel report error: " + str(e))

    if st.session_state.history:

        st.subheader("📈 Prediction History")

        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 16. ABOUT
# =========================================================

elif page == TEXT["about"]:

    st.title(TEXT["about"])

    st.markdown("""
    ## 🩺 HealthPredict AI 4.0

    An educational AI health and wellness project.

    ### ✨ Main Features

    - 🩺 Diabetes risk estimation using an existing ANN model
    - 🤖 Gemini-powered health education assistant
    - 🌐 Tamil and English interface
    - 🥗 Healthy food and nutrition guide
    - 🏃 Beginner exercise and fitness information
    - 📚 General health learning center
    - 📊 Prediction history
    - 📄 PDF health report
    - 📈 Excel report
    - 🎨 Multiple UI themes

    ### 🛠️ Technologies

    Python, Streamlit, TensorFlow, Keras, Scikit-learn,
    Google Gemini API, Pandas, NumPy, ReportLab and OpenPyXL.
    """)

    st.error(
        "MEDICAL DISCLAIMER: HealthPredict AI is not a hospital, doctor "
        "or diagnostic system. Its diabetes model estimates risk only. "
        "It does not diagnose diabetes or detect all diseases. Do not "
        "use its outputs to start, stop or change medication. Consult "
        "a qualified healthcare professional for medical decisions. "
        "For emergencies, contact local emergency services."
    )


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    "🩺 HealthPredict AI 4.0 | Health education and wellness | "
    "Not a substitute for professional medical care."
)
