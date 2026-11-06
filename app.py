
# ---------------- FUTURISTIC UI DESIGN ----------------

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
