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
# 2) 样式：更大字体 + 配色 + 卡片布局
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
  font-size: 20px !important;     /* ✅ 全局再大一档 */
  line-height: 1.6;
}

/* ✅ 统一放大 Markdown 文字（描述、普通段落等） */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p,
.stMarkdown li{
  font-size: 1.15rem !important;
}

/* ✅ 强制隐藏旧版本残留的 “Feature order ...” 行（就算你没删干净也不会显示） */
.small-note{
  display: none !important;
}

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
  font-size: 1.30rem;             /* ✅ 标题更大 */
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0 0 12px 0;
}

.desc{
  font-size: 1.22rem;             /* ✅ 描述更大 */
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
  font-size: 1.22rem;             /* ✅ 小标题更大 */
  font-weight: 800;
  margin: 0;
}

/* ✅ Radio 标题和选项都放大 */
.stRadio > label{
  font-weight: 800 !important;
  font-size: 1.18rem !important;
}
div[role="radiogroup"] label{
  font-size: 1.12rem !important;
  font-weight: 700 !important;
}

/* ✅ 输入标签更大 */
.stNumberInput label{
  font-size: 1.18rem !important;
  font-weight: 800 !important;
  color: #111827;
}

/* ✅ 输入框的数值更大 */
div[data-baseweb="input"] input{
  font-size: 1.18rem !important;
  padding: 12px 14px !important;
}

/* ✅ 按钮更大 */
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
  font-size: 1.38rem;             /* ✅ 结果更大 */
  font-weight: 900;
  margin: 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# 3) 加载模型
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("ada.pkl")

model = load_model()

# -------------------------
# 4) 语言切换 & 文本包（已去掉 Feature order 那句）
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
            "🧪 Adsorbent dosage (g/L)",
            "🌤 Temperature (°C)"
        ],
        "button_predict": "🔍 Predict adsorption capacity",
        "button_export": "📁 Export CSV",
        "result_prefix": "✅ Predicted TC adsorption capacity:",
        "file_name": "tc_prediction_result.csv",
        "section_inputs": "Input conditions"
    },
    "中文": {
        "title": "🔬 Fe@RSBC-β-CD 对四环素（TC）吸附量的机器学习预测",
        "description": "根据给定实验条件，预测 Fe@RSBC-β-CD 对四环素（TC）的单位吸附量（mg/g）。",
        # 按附件数据列顺序：C0 → Time → pH → Dosage → Temp
        "input_labels": [
            "💧 初始四环素浓度 C0 (mg/L)",
            "⏱ 吸附时间 (min)",
            "🌡 溶液 pH",
            "🧪 吸附剂投加量 (g/L)",
            "🌤 温度 (°C)"
        ],
        "button_predict": "🔍 预测吸附量",
        "button_export": "📁 导出 CSV",
        "result_prefix": "✅ 预测的四环素吸附量：",
        "file_name": "四环素预测结果.csv",
        "section_inputs": "输入条件"
    }
}[lang]

# -------------------------
# 5) 标题卡片（✅ 不再显示 Feature order 行）
# -------------------------
st.markdown(f"""
<div class="header-card">
  <div class="title">{text["title"]}</div>
  <p class="desc">{text["description"]}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# 6) 输入（按附件顺序：C0 → Time → pH → Dosage → Temp）
# -------------------------
st.markdown(f"""
<div class="input-card">
  <div class="section-title">🧩 {text["section_inputs"]}</div>
</div>
""", unsafe_allow_html=True)

c0 = st.number_input(text["input_labels"][0], min_value=0.0, value=50.0, step=1.0)
ads_time = st.number_input(text["input_labels"][1], min_value=0.0, value=120.0, step=1.0)
pH = st.number_input(text["input_labels"][2], min_value=1.0, max_value=14.0, value=7.0, step=0.1)
dosage = st.number_input(text["input_labels"][3], min_value=0.0, value=1.0, step=0.1)
temperature = st.number_input(text["input_labels"][4], min_value=0.0, value=25.0, step=1.0)

# -------------------------
# 7) 预测 + 导出
# -------------------------
prediction = None
df_result = None

if st.button(text["button_predict"]):
    # 模型输入顺序：C0, Time, pH, Dosage, Temp
    input_data = np.array([[c0, ads_time, pH, dosage, temperature]], dtype=float)
    prediction = float(model.predict(input_data)[0])

    st.markdown(
        f"""
        <div class="result-card">
          <p class="result-text">{text['result_prefix']} <span style="color:#15803d;">{prediction:.2f} mg/g</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 导出列顺序：C0 → Time → pH → Dosage → Temp
    df_result = pd.DataFrame([{
        "C0": c0,
        "Time": ads_time,
        "pH": pH,
        "Dosage": dosage,
        "Temp": temperature,
        "Predicted TC Adsorption (mg/g)": round(prediction, 2)
    }], columns=["C0", "Time", "pH", "Dosage", "Temp", "Predicted TC Adsorption (mg/g)"])

if prediction is not None and df_result is not None:
    towrite = BytesIO()
    df_result.to_csv(towrite, index=False)
    st.download_button(
        label=text["button_export"],
        data=towrite.getvalue(),
        file_name=text["file_name"],
        mime="text/csv"
    )




