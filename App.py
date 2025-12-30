import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# -------------------------
# 1) 页面配置
# -------------------------
st.set_page_config(
    page_title="TC Adsorption Prediction (Fe@RSBC-β-CD)",
    layout="centered"
)

# -------------------------
# 2) 样式（保持你现在的字体设置：英文标题不变）
# -------------------------
st.markdown("""
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

/* 强制隐藏旧版本残留的 “Feature order ...” 行 */
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
  font-size: 1.85rem;   /* ✅ 英文标题字体大小保持不变 */
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
""", unsafe_allow_html=True)

# -------------------------
# 3) 加载模型（✅ 解决“换模型文件但预测不变”的缓存问题）
#    - 用文件修改时间 mtime 作为 cache key
#    - 提供手动 Reload 按钮
# -------------------------
MODEL_PATH = "ada.pkl"

@st.cache_resource
def load_model(model_path: str, mtime: float):
    return joblib.load(model_path)

def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    mtime = os.path.getmtime(MODEL_PATH)
    return load_model(MODEL_PATH, mtime)

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("🔄 Reload model"):
        st.cache_resource.clear()
with col2:
    st.caption(f"Using model: {MODEL_PATH}")

model = get_model()

# -------------------------
# 4) 语言切换 & 文本包
# -------------------------
lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)

text = {
    "English": {
        "title": "🔬 ML prediction of tetracycline (TC) adsorption on Fe@RSBC-β-CD",
        "description": "Predict the TC adsorption capacity (mg/g) of Fe@RSBC-β-CD under specified experimental conditions.",
        "input_labels": [
            "💧 Initial TC concentration, C0 (mg/L)",
            "⏱ Adsorption time (min)",
            "🌡 Solution pH",
            "🧪 Adsorbent dosage (mg)",
            "🌤 Temperature (°C)"
        ],
        "button_predict": "🔍 Predict adsorption capacity",
        "button_export": "📁 Export CSV",
        "result_prefix": "✅ Predicted TC adsorption capacity:",
        "file_name": "tc_prediction_result.csv",
        "section_inputs": "Input conditions",
        "debug_title": "Debug (check inputs)"
    },
    "中文": {
        "title": "🔬 Fe@RSBC-β-CD 对四环素（TC）吸附量的机器学习预测",
        "description": "根据给定实验条件，预测 Fe@RSBC-β-CD 对四环素（TC）的单位吸附量（mg/g）。",
        "input_labels": [
            "💧 初始四环素浓度 C0 (mg/L)",
            "⏱ 吸附时间 (min)",
            "🌡 溶液 pH",
            "🧪 吸附剂投加量 (mg)",
            "🌤 温度 (°C)"
        ],
        "button_predict": "🔍 预测吸附量",
        "button_export": "📁 导出 CSV",
        "result_prefix": "✅ 预测的四环素吸附量：",
        "file_name": "四环素预测结果.csv",
        "section_inputs": "输入条件",
        "debug_title": "调试（检查输入）"
    }
}[lang]

# -------------------------
# 5) 标题 + 描述
# -------------------------
st.markdown(f"""
<div class="header-card">
  <div class="title">{text["title"]}</div>
  <p class="desc">{text["description"]}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# 6) 输入（顺序：C0 → Time → pH → Dosage → Temp）
# -------------------------
st.markdown(f"""
<div class="input-card">
  <div class="section-title">🧩 {text["section_inputs"]}</div>
</div>
""", unsafe_allow_html=True)

c0 = st.number_input(text["input_labels"][0], min_value=0.0, value=50.0, step=1.0)
ads_time = st.number_input(text["input_labels"][1], min_value=0.0, value=120.0, step=1.0)
pH = st.number_input(text["input_labels"][2], min_value=1.0, max_value=14.0, value=7.0, step=0.1)
dosage = st.number_input(text["input_labels"][3], min_value=0.0, value=20.0, step=1.0)
temperature = st.number_input(text["input_labels"][4], min_value=0.0, value=25.0, step=1.0)

input_data = np.array([[c0, ads_time, pH, dosage, temperature]], dtype=float)

# -------------------------
# 7) Debug：确认输入确实变了（不影响界面，折叠里看）
# -------------------------
with st.expander(f"🧾 {text['debug_title']}", expanded=False):
    st.write("Model type:", type(model))
    nfi = getattr(model, "n_features_in_", None)
    if nfi is not None:
        st.write("m
