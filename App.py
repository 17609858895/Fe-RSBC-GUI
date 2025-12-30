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
# 2) 样式：字体 + 配色 + 卡片布局
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg1:#f6f9ff;
  --bg2:#eef3ff;
  --card:#ffffffcc;
  --border: rgba(20, 40, 80, 0.10);
  --text:#1f2937;
  --muted:#6b7280;
  --accent:#3b82f6;
  --accent2:#22c55e;
}

.stApp{
  max-width: 720px;
  margin: 0 auto;
  padding: 2.2rem 1.8rem 3.2rem 1.8rem;
  background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}

html, body, [class*="css"]{
  font-family: 'Inter', 'Segoe UI', sans-serif;
  color: var(--text);
}

.block-container{
  padding-top: 0.6rem !important;
}

.header-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 18px 14px 18px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
  margin-bottom: 14px;
}

.title{
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 6px 0;
}

.desc{
  font-size: 1.02rem;
  color: var(--muted);
  margin: 0;
  line-height: 1.55;
}

.input-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px 18px 6px 18px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
  margin-top: 12px;
}

.section-title{
  font-weight: 650;
  margin: 0 0 10px 0;
  color: #111827;
}

.stRadio > label{
  font-weight: 600;
}

.stNumberInput label{
  font-size: 0.98rem;
  font-weight: 600;
  color: #111827;
}

.stButton > button{
  width: 100%;
  background: var(--accent);
  color: white;
  font-weight: 700;
  font-size: 1.02rem;
  padding: 0.62rem 1.0rem;
  border-radius: 12px;
  border: none;
  margin-top: 14px;
  box-shadow: 0 10px 18px rgba(59,130,246,0.18);
}

.stButton > button:hover{
  filter: brightness(0.98);
  transform: translateY(-1px);
}

.stDownloadButton > button{
  width: 100%;
  background: white;
  color: var(--text);
  font-weight: 650;
  border: 1px solid rgba(31,41,55,0.14);
  border-radius: 12px;
  margin-top: 10px;
  padding: 0.55rem 1.0rem;
}

.result-card{
  background: rgba(34,197,94,0.10);
  border: 1px solid rgba(34,197,94,0.22);
  border-radius: 16px;
  padding: 14px 18px;
  margin-top: 14px;
}

.result-text{
  font-size: 1.08rem;
  font-weight: 700;
  margin: 0;
}

.small-note{
  font-size: 0.92rem;
  color: var(--muted);
  margin-top: 6px;
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
# 4) 语言切换 & 文本包
# -------------------------
lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)

text = {
    "English": {
        "title": "🔬 ML prediction of tetracycline (TC) adsorption on Fe@RSBC-β-CD",
        "description": "Predict the TC adsorption capacity (mg/g) of Fe@RSBC-β-CD under specified experimental conditions.",
        # 按附件数据列顺序：C0 → Time → pH → Dosage → Temp
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
        "section_inputs": "Input conditions",
        "note_order": "Feature order (model input): C0 → Time → pH → Dosage → Temp"
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
        "section_inputs": "输入条件",
        "note_order": "模型特征顺序：C0 → Time → pH → Dosage → Temp"
    }
}[lang]

# -------------------------
# 5) 标题卡片
# -------------------------
st.markdown(f"""
<div class="header-card">
  <div class="title">{text["title"]}</div>
  <p class="desc">{text["description"]}</p>
  <p class="small-note">🧾 {text["note_order"]}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# 6) 输入（按附件顺序）
# -------------------------
st.markdown(f"""
<div class="input-card">
  <div class="section-title">🧩 {text["section_inputs"]}</div>
</div>
""", unsafe_allow_html=True)

# 注意：显示顺序与传入顺序一致：C0 → Time → pH → Dosage → Temp
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
    # 传入模型的特征顺序：C0, Time, pH, Dosage, Temp（与附件 data.xlsx 一致）
    input_data = np.array([[c0, ads_time, pH, dosage, temperature]], dtype=float)
    prediction = float(model.predict(input_data)[0])

    st.markdown(
        f"""
        <div class="result-card">
          <p class="result-text">{text["result_prefix"]} <span style="color:#15803d;">{prediction:.2f} mg/g</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 导出列顺序：C0 → Time → pH → Dosage → Temp（与附件一致）
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
