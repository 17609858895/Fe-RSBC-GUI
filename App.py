"""Streamlit interface for the final Extra Trees TC-adsorption model."""

from __future__ import annotations

import os
from io import BytesIO

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "et_model_bundle.joblib")
EXPECTED_FEATURES = ["C0", "Time", "pH", "Dosage", "Temp"]


st.set_page_config(
    page_title="TC adsorption prediction | Fe@RSBC-β-CD",
    page_icon="🔬",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
      --navy: #17324d;
      --blue: #2f6f9f;
      --pale: #edf5fa;
      --line: rgba(23, 50, 77, 0.14);
      --muted: #536779;
    }
    html, body, [class*="css"] {
      font-family: 'Inter', 'Segoe UI', sans-serif;
      color: var(--navy);
    }
    .stApp {
      max-width: 900px;
      margin: 0 auto;
      background: linear-gradient(180deg, #f8fbfd 0%, #eef5f9 100%);
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .hero, .info-card, .result-card {
      background: rgba(255,255,255,0.94);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 28px rgba(23,50,77,0.07);
    }
    .hero { padding: 24px 26px 20px 26px; margin-bottom: 16px; }
    .hero h1 { font-size: 1.72rem; margin: 0 0 10px 0; line-height: 1.25; }
    .hero p { color: var(--muted); font-size: 1.02rem; margin: 0; line-height: 1.65; }
    .info-card { padding: 14px 18px; margin: 12px 0 18px 0; }
    .info-card p { margin: 3px 0; color: var(--muted); font-size: 0.90rem; }
    .result-card {
      padding: 20px 22px;
      margin-top: 18px;
      background: linear-gradient(135deg, #edf8f2 0%, #f7fcf9 100%);
      border-color: rgba(34, 125, 83, 0.22);
    }
    .result-label { color: #426352; font-size: 0.94rem; margin-bottom: 5px; }
    .result-value { color: #17633f; font-size: 2rem; font-weight: 800; margin: 0; }
    .stNumberInput label { font-weight: 700 !important; }
    .stButton > button {
      width: 100%; border-radius: 12px; border: 0;
      background: var(--blue); color: white; font-weight: 800;
      min-height: 3rem; margin-top: 8px;
    }
    .stDownloadButton > button { width: 100%; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(path: str, modified_time: float):
    del modified_time
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError("The model bundle is incomplete.")
    metadata = bundle["metadata"]
    if metadata.get("feature_order") != EXPECTED_FEATURES:
        raise ValueError("The deployed feature order does not match the validated model.")
    return bundle["model"], metadata


try:
    model, metadata = load_bundle(MODEL_PATH, os.path.getmtime(MODEL_PATH))
except Exception as exc:
    st.error(str(exc))
    st.stop()


language = st.radio("Language / 语言", ["English", "中文"], horizontal=True)

TEXT = {
    "English": {
        "title": "Machine-learning prediction of TC adsorption on Fe@RSBC-β-CD",
        "description": (
            "Estimate tetracycline (TC) adsorption capacity under conditions within "
            "the experimentally investigated domain."
        ),
        "model": "Validated model",
        "evidence": "Validation evidence",
        "scope": "Use boundary",
        "inputs": "Experimental conditions",
        "predict": "Predict adsorption capacity",
        "result": "Predicted TC adsorption capacity",
        "download": "Export prediction as CSV",
        "caution": (
            "Research-use decision support only. The dataset is small, validation "
            "performance varies across splits, and extrapolation beyond the displayed "
            "ranges has not been established."
        ),
        "labels": {
            "C0": "Initial TC concentration, C₀ (mg L⁻¹)",
            "Time": "Adsorption time (min)",
            "pH": "Solution pH",
            "Dosage": "Adsorbent dosage (mg)",
            "Temp": "Temperature (°C)",
        },
    },
    "中文": {
        "title": "Fe@RSBC-β-CD 对四环素吸附量的机器学习预测",
        "description": "在实验考察范围内，根据给定条件估算四环素（TC）吸附量。",
        "model": "验证后的模型",
        "evidence": "验证依据",
        "scope": "适用边界",
        "inputs": "实验条件",
        "predict": "预测吸附量",
        "result": "预测的四环素吸附量",
        "download": "导出预测结果（CSV）",
        "caution": (
            "仅用于科研辅助判断。由于数据量有限且不同划分下的验证性能存在波动，"
            "本工具不支持超出下列实验范围的外推。"
        ),
        "labels": {
            "C0": "初始四环素浓度 C₀ (mg L⁻¹)",
            "Time": "吸附时间 (min)",
            "pH": "溶液 pH",
            "Dosage": "吸附剂投加量 (mg)",
            "Temp": "温度 (°C)",
        },
    },
}[language]


st.markdown(
    f"""
    <div class="hero">
      <h1>🔬 {TEXT['title']}</h1>
      <p>{TEXT['description']}</p>
    </div>
    <div class="info-card">
      <p><strong>{TEXT['model']}:</strong> Extra Trees (ET)</p>
      <p><strong>{TEXT['evidence']}:</strong> 20 group-aware random splits and 10 × 5-fold nested group cross-validation</p>
      <p><strong>{TEXT['scope']}:</strong> {metadata['scope']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader(TEXT["inputs"])
bounds = metadata["feature_bounds"]
steps = {"C0": 1.0, "Time": 1.0, "pH": 0.1, "Dosage": 1.0, "Temp": 1.0}
values: dict[str, float] = {}

left, right = st.columns(2)
for index, feature in enumerate(EXPECTED_FEATURES):
    target_column = left if index % 2 == 0 else right
    spec = bounds[feature]
    with target_column:
        values[feature] = st.number_input(
            TEXT["labels"][feature],
            min_value=float(spec["minimum"]),
            max_value=float(spec["maximum"]),
            value=float(spec["median"]),
            step=steps[feature],
        )

domain_text = " · ".join(
    f"{feature}: {bounds[feature]['minimum']:g}–{bounds[feature]['maximum']:g}"
    for feature in EXPECTED_FEATURES
)
st.caption(domain_text)

prediction = None
if st.button(TEXT["predict"], type="primary"):
    # Construct the frame in the exact order stored with the validated model.
    model_input = pd.DataFrame(
        [[values[feature] for feature in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES,
    )
    prediction = float(model.predict(model_input.to_numpy(dtype=float))[0])
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">{TEXT['result']}</div>
          <p class="result-value">{prediction:.2f} mg g⁻¹</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if prediction is not None:
    output = {feature: values[feature] for feature in EXPECTED_FEATURES}
    output["Predicted TC adsorption capacity (mg g-1)"] = round(prediction, 4)
    buffer = BytesIO()
    pd.DataFrame([output]).to_csv(buffer, index=False)
    st.download_button(
        TEXT["download"],
        data=buffer.getvalue(),
        file_name="Fe_RSBC_TC_prediction.csv",
        mime="text/csv",
    )

st.warning(TEXT["caution"], icon="⚠️")
