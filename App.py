import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# 页面配置
st.set_page_config(
    page_title="TC Adsorption Prediction (Ca@CBC/β-CD)",
    layout="centered"
)

# 轻量样式（浅蓝背景）
st.markdown("""
    <style>
    .stApp {
        max-width: 610px;
        margin: auto;
        background-color: #eef6ff; /* light blue */
        padding: 2rem 2rem 4rem 2rem;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .custom-title {
        font-size: 1.8rem;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 0.1rem;
        color: #222;
    }
    .stMarkdown h1 + p {
        font-size: 1.02rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .stNumberInput label {
        font-size: 0.98rem;
        font-weight: 500;
        color: #333;
    }
    .stButton>button {
        background-color: #4caf91;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.55rem 1.1rem;
        border-radius: 10px;
        border: none;
        margin-top: 1.3rem;
    }
    .stDownloadButton>button {
        background-color: white;
        color: #333;
        font-weight: 500;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-top: 1rem;
        padding: 0.5rem 1rem;
    }
    .stSuccess {
        background-color: #e6f9ed;
        color: #1b5e20;
        padding: 0.85rem;
        border-radius: 8px;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load("Catboost.pkl")

model = load_model()

# 语言切换
lang = st.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)

# 文本包（已统一为 Ca@CBC/β-CD & tetracycline/四环素）
text = {
    "English": {
        "title": "🔬 ML prediction of tetracycline (TC) adsorption on Ca@CBC/β-CD",
        "description": "Predict the tetracycline (TC) adsorption capacity (mg/g) of Ca@CBC/β-CD under specified experimental conditions.",
        # 输入标签顺序：pH → C0 → Dosage → Temp → Time
        "input_labels": [
            "🌡 Solution pH",
            "💧 Initial TC Concentration, C0 (mg/L)",
            "🧪 Adsorbent Dosage (g/L)",
            "🌤 Temperature (°C)",
            "⏱ Adsorption Time (min)"
        ],
        "button_predict": "🔍 Predict Adsorption Capacity",
        "button_export": "📁 Export CSV",
        "result_prefix": "✅ Predicted TC adsorption capacity:",
        "file_name": "tc_prediction_result.csv"
    },
    "中文": {
        "title": "🔬 Ca@CBC/β-CD 对四环素（TC）吸附量的机器学习预测",
        "description": "根据给定实验条件，预测 Ca@CBC/β-CD 对四环素（TC）的单位吸附量（mg/g）。",
        # 输入标签顺序：pH → C0 → Dosage → Temp → Time
        "input_labels": [
            "🌡 溶液 pH",
            "💧 初始四环素浓度 C0 (mg/L)",
            "🧪 吸附剂投加量 (g/L)",
            "🌤 温度 (°C)",
            "⏱ 吸附时间 (分钟)"
        ],
        "button_predict": "🔍 预测吸附量",
        "button_export": "📁 导出 CSV",
        "result_prefix": "✅ 预测的四环素吸附量：",
        "file_name": "四环素预测结果.csv"
    }
}[lang]

# 标题 + 描述
st.markdown(f'<div class="custom-title">{text["title"]}</div>', unsafe_allow_html=True)
st.markdown(text["description"])

# 输入字段 —— 显示与变量顺序均为：pH → C0 → Dosage → Temp → Time
pH = st.number_input(text["input_labels"][0], min_value=1.0, max_value=14.0, value=7.0, step=0.1)
c0 = st.number_input(text["input_labels"][1], min_value=0.0, value=50.0, step=1.0)
dosage = st.number_input(text["input_labels"][2], min_value=0.0, value=1.0, step=0.1)
temperature = st.number_input(text["input_labels"][3], min_value=0.0, value=25.0, step=1.0)
ads_time = st.number_input(text["input_labels"][4], min_value=0.0, value=120.0, step=1.0)

# 预测
prediction = None
df_result = None

if st.button(text["button_predict"]):
    # 传入模型的特征顺序：pH, C0, Dosage, Temp, Time（与数据集列顺序一致）
    input_data = np.array([[pH, c0, dosage, temperature, ads_time]], dtype=float)
    prediction = model.predict(input_data)[0]
    st.success(f"{text['result_prefix']} **{prediction:.2f} mg/g**")

    # 导出表头：pH, C0, Dosage, Temp, Time（与截图一致；不含单位）
    df_result = pd.DataFrame([{
        "pH": pH,
        "C0": c0,
        "Dosage": dosage,
        "Temp": temperature,
        "Time": ads_time,
        "Predicted TC Adsorption (mg/g)": round(prediction, 2)
    }], columns=["pH", "C0", "Dosage", "Temp", "Time", "Predicted TC Adsorption (mg/g)"])

# 导出 CSV
if prediction is not None and df_result is not None:
    towrite = BytesIO()
    df_result.to_csv(towrite, index=False)
    st.download_button(
        label=text["button_export"],
        data=towrite.getvalue(),
        file_name=text["file_name"],
        mime="text/csv"
    )
