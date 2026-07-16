"""Streamlit interface for the validated Extra Trees TC-adsorption model."""

from __future__ import annotations

import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "et_model_bundle.joblib")
FEATURES = ["C0", "Time", "pH", "Dosage", "Temp"]


st.set_page_config(
    page_title="TC Adsorption Prediction (Fe@RSBC-β-CD)",
    layout="centered",
)


# Keep the original visual system while using the revised validated model.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg1:#f6f9ff;
  --bg2:#eef3ff;
  --card:#ffffffcc;
  --border: rgba(20, 40, 80, 0.10);
  --text:#111827;
  --muted:#4b5563;
  --accent:#3b82f6;
}

.stApp{
  max-width: 820px;
  margin: 0 auto;
  padding: 2.6rem 2.2rem 3.8rem 2.2rem;
  background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}

html, body, [class*="css"]{
  font-family: 'Inter', 'Segoe UI', sans-serif;
  color: var(--text);
  font-size: 20px !important;
  line-height: 1.6;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p,
.stMarkdown li{
  font-size: 1.15rem !important;
}

.small-note{ display: none !important; }

.block-container{
  padding-top: 0.6rem !important;
}

.header-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 22px 22px 18px 22px;
  box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
  margin-bottom: 18px;
}

.title{
  font-size: 1.85rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0 0 12px 0;
}

.desc{
  font-size: 1.22rem;
  color: var(--muted);
  margin: 0;
  line-height: 1.7;
}

.input-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 20px 8px 20px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
  margin-top: 14px;
  margin-bottom: 10px;
}

.section-title{
  font-size: 1.22rem;
  font-weight: 800;
  margin: 0;
}

.stRadio > label{
  font-weight: 800 !important;
  font-size: 1.18rem !important;
}
div[role="radiogroup"] label{
  font-size: 1.12rem !important;
  font-weight: 700 !important;
}

.stNumberInput label{
  font-size: 1.18rem !important;
  font-weight: 800 !important;
  color: #111827;
}

div[data-baseweb="input"] input{
  font-size: 1.18rem !important;
  padding: 12px 14px !important;
}

.stButton > button{
  width: 100%;
  background: var(--accent);
  color: white;
  font-weight: 900;
  font-size: 1.22rem !important;
  padding: 0.9rem 1.2rem;
  border-radius: 14px;
  border: none;
  margin-top: 18px;
  box-shadow: 0 12px 20px rgba(59,130,246,0.18);
}

.stDownloadButton > button{
  width: 100%;
  background: white;
  color: var(--text);
  font-weight: 800;
  font-size: 1.15rem !important;
  border: 1px solid rgba(31,41,55,0.14);
  border-radius: 14px;
  margin-top: 12px;
  padding: 0.85rem 1.2rem;
}

.result-card{
  background: rgba(34,197,94,0.10);
  border: 1px solid rgba(34,197,94,0.22);
  border-radius: 18px;
  padding: 18px 22px;
  margin-top: 18px;
}

.result-text{
  font-size: 1.38rem;
  font-weight: 900;
  margin: 0;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets(model_path: str, modified_time: float):
    del modified_time
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError("The model bundle is incomplete.")

    metadata = bundle["metadata"]
    if metadata.get("feature_order") != FEATURES:
        raise ValueError("The deployed feature order does not match the validated model.")
    return bundle["model"], metadata


def get_assets():
    return load_assets(MODEL_PATH, os.path.getmtime(MODEL_PATH))


reload_column, model_column = st.columns([1, 3])
with reload_column:
    if st.button("🔄 Reload"):
        st.cache_resource.clear()
with model_column:
    st.caption("Using: et_model_bundle.joblib (Extra Trees)")

try:
    model, metadata = get_assets()
except Exception as exc:
    st.error(str(exc))
    st.stop()


lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)

text = {
    "English": {
        "title": "🔬 ML prediction of tetracycline (TC) adsorption on Fe@RSBC-β-CD",
        "description": (
            "Predict the TC adsorption capacity (mg g⁻¹) of Fe@RSBC-β-CD "
            "under specified experimental conditions."
        ),
        "input_labels": [
            "💧 Initial TC concentration, C₀ (mg L⁻¹)",
            "⏱ Adsorption time (min)",
            "🌡 Solution pH",
            "🧪 Adsorbent dosage (mg)",
            "🌤 Temperature (°C)",
        ],
        "button_predict": "🔍 Predict adsorption capacity",
        "button_export": "📁 Export CSV",
        "result_prefix": "✅ Predicted TC adsorption capacity:",
        "file_name": "tc_prediction_result.csv",
        "section_inputs": "Input conditions",
        "debug_title": "Debug / sanity check",
    },
    "中文": {
        "title": "🔬 Fe@RSBC-β-CD 对四环素（TC）吸附量的机器学习预测",
        "description": "根据给定实验条件，预测 Fe@RSBC-β-CD 对四环素（TC）的单位吸附量（mg g⁻¹）。",
        "input_labels": [
            "💧 初始四环素浓度 C₀ (mg L⁻¹)",
            "⏱ 吸附时间 (min)",
            "🌡 溶液 pH",
            "🧪 吸附剂投加量 (mg)",
            "🌤 温度 (°C)",
        ],
        "button_predict": "🔍 预测吸附量",
        "button_export": "📁 导出 CSV",
        "result_prefix": "✅ 预测的四环素吸附量：",
        "file_name": "四环素预测结果.csv",
        "section_inputs": "输入条件",
        "debug_title": "调试 / 自检",
    },
}[lang]


st.markdown(
    f"""
<div class="header-card">
  <div class="title">{text["title"]}</div>
  <p class="desc">{text["description"]}</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="input-card">
  <div class="section-title">🧩 {text["section_inputs"]}</div>
</div>
""",
    unsafe_allow_html=True,
)

bounds = metadata["feature_bounds"]
c0 = st.number_input(
    text["input_labels"][0],
    min_value=float(bounds["C0"]["minimum"]),
    max_value=float(bounds["C0"]["maximum"]),
    value=float(bounds["C0"]["median"]),
    step=1.0,
)
ads_time = st.number_input(
    text["input_labels"][1],
    min_value=float(bounds["Time"]["minimum"]),
    max_value=float(bounds["Time"]["maximum"]),
    value=float(bounds["Time"]["median"]),
    step=1.0,
)
pH = st.number_input(
    text["input_labels"][2],
    min_value=float(bounds["pH"]["minimum"]),
    max_value=float(bounds["pH"]["maximum"]),
    value=float(bounds["pH"]["median"]),
    step=0.1,
)
dosage = st.number_input(
    text["input_labels"][3],
    min_value=float(bounds["Dosage"]["minimum"]),
    max_value=float(bounds["Dosage"]["maximum"]),
    value=float(bounds["Dosage"]["median"]),
    step=1.0,
)
temperature = st.number_input(
    text["input_labels"][4],
    min_value=float(bounds["Temp"]["minimum"]),
    max_value=float(bounds["Temp"]["maximum"]),
    value=float(bounds["Temp"]["median"]),
    step=1.0,
)

raw_input = np.array([[c0, ads_time, pH, dosage, temperature]], dtype=float)

prediction = None
df_result = None
if st.button(text["button_predict"]):
    prediction = float(model.predict(raw_input)[0])
    st.markdown(
        f"""
        <div class="result-card">
          <p class="result-text">{text['result_prefix']} <span style="color:#15803d;">{prediction:.2f} mg g⁻¹</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_result = pd.DataFrame(
        [
            {
                "C₀ (mg L⁻¹)": c0,
                "Time (min)": ads_time,
                "pH": pH,
                "Dosage (mg)": dosage,
                "Temperature (°C)": temperature,
                "Predicted TC adsorption capacity (mg g⁻¹)": round(prediction, 2),
            }
        ],
        columns=[
            "C₀ (mg L⁻¹)",
            "Time (min)",
            "pH",
            "Dosage (mg)",
            "Temperature (°C)",
            "Predicted TC adsorption capacity (mg g⁻¹)",
        ],
    )

if prediction is not None and df_result is not None:
    towrite = BytesIO()
    df_result.to_csv(towrite, index=False)
    st.download_button(
        label=text["button_export"],
        data=towrite.getvalue(),
        file_name=text["file_name"],
        mime="text/csv",
    )

with st.expander(f"🧾 {text['debug_title']}", expanded=False):
    st.write("Model type:", type(model))
    display_feature_order = ["C₀", "Time", "pH", "Dosage", "Temp"]
    st.write("Feature order:", display_feature_order)
    st.write("Current raw input (C₀, Time, pH, Dosage, Temp):")
    st.code(str(raw_input))

    sample_a = np.array([[40, 120, 7, 20, 25]], dtype=float)
    sample_b = np.array([[100, 120, 7, 20, 25]], dtype=float)
    prediction_a = float(model.predict(sample_a)[0])
    prediction_b = float(model.predict(sample_b)[0])
    st.write("Sanity check predictions (should differ):")
    st.write(
        {
            "[40,120,7,20,25]": prediction_a,
            "[100,120,7,20,25]": prediction_b,
        }
    )
