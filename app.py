import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DenguePredict AI",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1c2333;
    --border:    #2a3347;
    --accent:    #f97316;
    --accent2:   #fb923c;
    --danger:    #ef4444;
    --success:   #22c55e;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    12px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Header ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(249,115,22,0.18) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #fb923c, #f97316, #fdba74);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
}
.badge {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    color: var(--accent2);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Section cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
}

/* ── Sliders & inputs ── */
div[data-baseweb="slider"] > div {
    background: var(--accent) !important;
}
div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #ea580c, #f97316) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(249,115,22,0.35) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(249,115,22,0.5) !important;
}

/* ── Result box ── */
.result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.05));
    border: 1.5px solid rgba(239,68,68,0.5);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.05));
    border: 1.5px solid rgba(34,197,94,0.5);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    text-align: center;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.3rem 0;
}
.result-score {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.4rem;
}
.prob-bar-wrap {
    background: var(--surface2);
    border-radius: 999px;
    height: 8px;
    margin: 1rem 0 0.4rem;
    overflow: hidden;
}
.prob-bar-inner {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Info grid ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1rem;
}
.info-cell {
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
}
.info-cell-label { color: var(--muted); margin-bottom: 0.2rem; }
.info-cell-val   { font-weight: 600; font-size: 0.95rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(249,115,22,0.07);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 1.5rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Model (trained on synthetic but clinically-informed data) ─────────────────
@st.cache_resource
def build_model():
    rng = np.random.default_rng(42)
    n = 2000

    fever      = rng.uniform(37.5, 41.5, n)
    platelet   = rng.uniform(20,  350,  n)
    wbc        = rng.uniform(1,   11,   n)
    hematocrit = rng.uniform(30,  60,   n)
    ns1        = rng.integers(0, 2, n)
    rash       = rng.integers(0, 2, n)
    headache   = rng.integers(0, 2, n)
    myalgia    = rng.integers(0, 2, n)
    retro_pain = rng.integers(0, 2, n)
    duration   = rng.uniform(1,  14,   n)

    # Clinically-motivated label generation
    score = (
        (fever > 38.5).astype(float) * 1.5 +
        (platelet < 100).astype(float) * 2.0 +
        (wbc < 4).astype(float) * 1.5 +
        (hematocrit > 45).astype(float) * 1.0 +
        ns1 * 2.5 +
        rash * 1.0 +
        headache * 0.8 +
        myalgia * 0.8 +
        retro_pain * 1.2 +
        (duration > 3).astype(float) * 0.5
    )
    labels = (score + rng.normal(0, 0.5, n) > 5).astype(int)

    X = np.column_stack([fever, platelet, wbc, hematocrit,
                         ns1, rash, headache, myalgia, retro_pain, duration])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    clf.fit(X_sc, labels)
    return clf, scaler

model, scaler = build_model()

FEATURES = [
    "Body Temperature (°C)",
    "Platelet Count (×10³/µL)",
    "WBC Count (×10³/µL)",
    "Hematocrit (%)",
    "NS1 Antigen",
    "Skin Rash",
    "Severe Headache",
    "Muscle / Joint Pain",
    "Retro-orbital Pain",
    "Fever Duration (days)",
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦟 About DenguePredict")
    st.markdown("""
This tool uses a **Random Forest** classifier trained on
clinically-informed synthetic data to estimate dengue fever risk
from common clinical indicators.

**Key features used:**
- Haematological markers
- NS1 antigen result
- Reported symptoms
- Fever duration

---
**References**
- WHO Dengue Guidelines (2009, 2012)
- CDC Clinical Guidance
---
""")
    st.markdown('<div class="disclaimer">⚠️ <b>Not a diagnostic tool.</b> This application is for educational and screening purposes only. Always consult a qualified healthcare professional for clinical diagnosis and treatment.</div>', unsafe_allow_html=True)


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">🦟 AI-Powered Screening</div>
  <div class="hero-title">DenguePredict AI</div>
  <p class="hero-sub">
    Enter clinical indicators below to receive an instant dengue fever
    risk assessment powered by a machine-learning classifier trained on
    haematological and symptomatic data.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # — Vital Signs —
    st.markdown('<div class="section-card"><div class="section-title">📊 Haematological Markers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fever      = st.slider("Body Temperature (°C)",    37.0, 42.0, 38.8, 0.1)
        platelet   = st.slider("Platelet Count (×10³/µL)", 10,   400,  90,   1)
    with c2:
        wbc        = st.slider("WBC Count (×10³/µL)",      1.0,  12.0, 3.5,  0.1)
        hematocrit = st.slider("Hematocrit (%)",           25,   65,   46,   1)
    st.markdown("</div>", unsafe_allow_html=True)

    # — Lab Test —
    st.markdown('<div class="section-card"><div class="section-title">🧪 Laboratory Test</div>', unsafe_allow_html=True)
    ns1 = st.radio(
        "NS1 Antigen Result",
        options=[0, 1],
        format_func=lambda x: "✅ Negative" if x == 0 else "🔴 Positive",
        horizontal=True,
    )
    duration = st.slider("Fever Duration (days)", 1, 14, 4)
    st.markdown("</div>", unsafe_allow_html=True)

    # — Symptoms —
    st.markdown('<div class="section-card"><div class="section-title">🩺 Reported Symptoms</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        rash       = int(st.checkbox("Skin Rash",             value=True))
        headache   = int(st.checkbox("Severe Headache",       value=True))
    with s2:
        myalgia    = int(st.checkbox("Muscle / Joint Pain",   value=True))
        retro_pain = int(st.checkbox("Retro-orbital Pain",    value=False))
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔍  Run Prediction")


# ── Results panel ─────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title" style="padding-top:0.2rem">📋 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        X_input = np.array([[fever, platelet, wbc, hematocrit,
                              ns1, rash, headache, myalgia, retro_pain, duration]])
        X_scaled = scaler.transform(X_input)
        proba    = model.predict_proba(X_scaled)[0]
        pred     = model.predict(X_scaled)[0]
        pos_prob = proba[1] * 100
        neg_prob = proba[0] * 100

        if pred == 1:
            bar_color = "#ef4444"
            result_class = "result-positive"
            icon  = "🔴"
            label = "Dengue POSITIVE"
            note  = "High risk detected. Immediate clinical evaluation is strongly advised."
        else:
            bar_color = "#22c55e"
            result_class = "result-negative"
            icon  = "🟢"
            label = "Dengue NEGATIVE"
            note  = "Low risk based on current indicators. Monitor symptoms closely."

        st.markdown(f"""
        <div class="{result_class}">
          <div style="font-size:2.2rem">{icon}</div>
          <div class="result-label">{label}</div>
          <div class="result-score">Confidence score</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-inner" style="width:{pos_prob:.1f}%;background:{bar_color}"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b">
            <span>Negative {neg_prob:.1f}%</span><span>Positive {pos_prob:.1f}%</span>
          </div>
          <p style="margin-top:1rem;font-size:0.82rem;color:#94a3b8">{note}</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance summary
        importances = model.feature_importances_
        feat_names  = ["Temperature", "Platelets", "WBC", "Hematocrit",
                       "NS1", "Rash", "Headache", "Myalgia", "Retro Pain", "Duration"]
        top_idx = np.argsort(importances)[::-1][:4]

        st.markdown("<br><div class='section-title'>🔑 Top Contributing Factors</div>", unsafe_allow_html=True)
        for i in top_idx:
            pct = importances[i] * 100
            st.markdown(f"""
            <div style="margin-bottom:0.6rem">
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.25rem">
                <span style="color:#e2e8f0">{feat_names[i]}</span>
                <span style="color:#64748b">{pct:.1f}%</span>
              </div>
              <div class="prob-bar-wrap">
                <div class="prob-bar-inner" style="width:{pct*4:.0f}%;background:var(--accent)"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary table
        st.markdown("<br><div class='section-title'>📥 Input Summary</div>", unsafe_allow_html=True)
        summary = {
            "Temperature": f"{fever} °C",
            "Platelets":   f"{platelet} ×10³/µL",
            "WBC":         f"{wbc} ×10³/µL",
            "Hematocrit":  f"{hematocrit}%",
            "NS1 Antigen": "Positive" if ns1 else "Negative",
            "Fever Days":  f"{duration} days",
        }
        df_summary = pd.DataFrame(summary.items(), columns=["Indicator", "Value"])
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div style="background:#111827;border:1px dashed #2a3347;border-radius:12px;
                    padding:3rem 2rem;text-align:center;color:#475569">
          <div style="font-size:3rem;margin-bottom:1rem">🦟</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                      color:#64748b;letter-spacing:0.05em">
            Fill in the clinical indicators<br>and click <b style="color:#f97316">Run Prediction</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #1e293b;margin:2.5rem 0 1rem">
<div style="text-align:center;font-size:0.75rem;color:#334155">
  DenguePredict AI · Built with Streamlit & scikit-learn ·
  <span style="color:#f97316">For educational use only</span>
</div>
""", unsafe_allow_html=True)
