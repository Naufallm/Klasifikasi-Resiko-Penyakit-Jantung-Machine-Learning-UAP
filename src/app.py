import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

st.set_page_config(page_title="HeartHealth AI", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #ff4b4b; font-weight: bold; border: none; color: white; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #ccd6f6 !important; font-weight: 500; }
    .result-card { padding: 25px; border-radius: 15px; background-color: #1e2129; border: 1px solid #30363d; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

class BaseMLP(nn.Module):
    def __init__(self, input_dim):
        super(BaseMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )
    def forward(self, x): return self.model(x)

class PretrainedStyleNN(nn.Module):
    def __init__(self, input_dim):
        super(PretrainedStyleNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Linear(64, 2)
    def forward(self, x): return self.classifier(self.feature_extractor(x))

@st.cache_resource
def load_assets():
    scaler = joblib.load('src/models/scaler.pkl')
    feature_names = joblib.load('src/models/feature_names.pkl')
    encoders = joblib.load('src/models/label_encoders.pkl')
    
    mlp = BaseMLP(len(feature_names))
    mlp.load_state_dict(torch.load('src/models/mlp_model.pth', map_location='cpu'))
    mlp.eval()
    
    pt_style = PretrainedStyleNN(len(feature_names))
    pt_style.load_state_dict(torch.load('src/models/pt_style_model.pth', map_location='cpu'))
    pt_style.eval()
    
    tabnet = TabNetClassifier()
    tabnet.load_model('src/models/tabnet_model.zip')
    
    return scaler, feature_names, encoders, mlp, pt_style, tabnet

try:
    scaler, feature_names, encoders, mlp_model, pt_style_model, tabnet_model = load_assets()
except Exception as e:
    st.error(f"Asset Error: {e}. Pastikan Anda sudah menjalankan kode ekspor terbaru di Notebook.")
    st.stop()

st.title("❤️ Heart Attack Risk Analyzer")
st.markdown("Analisis risiko serangan jantung menggunakan Artificial Intelligence.")

input_user = {}

st.subheader("📊 Profil Kesehatan")
col1, col2 = st.columns(2)

with col1:
    input_user['Sex'] = st.selectbox("Jenis Kelamin", encoders['Sex'].classes_) if 'Sex' in encoders else st.selectbox("Jenis Kelamin", ["Male", "Female"])
    input_user['AgeCategory'] = st.selectbox("Kategori Umur", encoders['AgeCategory'].classes_) if 'AgeCategory' in encoders else st.selectbox("Kategori Umur", ["Age 18 to 24", "Age 80 or older"])
    input_user['GeneralHealth'] = st.selectbox("Kesehatan Umum", encoders['GeneralHealth'].classes_) if 'GeneralHealth' in encoders else st.selectbox("Kesehatan Umum", ["Excellent", "Good", "Poor"])
    input_user['SmokerStatus'] = st.selectbox("Status Merokok", encoders['SmokerStatus'].classes_) if 'SmokerStatus' in encoders else st.selectbox("Status Merokok", ["Never smoked", "Smoker"])

with col2:
    height = st.number_input("Tinggi Badan (m)", 1.0, 2.5, 1.70)
    weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0)
    input_user['HeightInMeters'] = height
    input_user['WeightInKilograms'] = weight
    input_user['BMI'] = weight / (height ** 2)
    input_user['SleepHours'] = st.slider("Jam Tidur Harian", 1, 24, 7)
    input_user['PhysicalActivities'] = st.selectbox("Aktif Berolahraga?", encoders['PhysicalActivities'].classes_) if 'PhysicalActivities' in encoders else st.selectbox("Olahraga?", ["Yes", "No"])

st.subheader("🏥 Riwayat Medis")
col3, col4 = st.columns(2)

with col3:
    input_user['HadDiabetes'] = st.selectbox("Diabetes?", encoders['HadDiabetes'].classes_) if 'HadDiabetes' in encoders else st.selectbox("Diabetes?", ["No", "Yes"])
    input_user['HadStroke'] = st.selectbox("Pernah Stroke?", encoders['HadStroke'].classes_) if 'HadStroke' in encoders else st.selectbox("Stroke?", ["No", "Yes"])
    input_user['HadAsthma'] = st.selectbox("Asma?", encoders['HadAsthma'].classes_) if 'HadAsthma' in encoders else st.selectbox("Asma?", ["No", "Yes"])

with col4:
    input_user['AlcoholDrinkers'] = st.selectbox("Peminum Alkohol?", encoders['AlcoholDrinkers'].classes_) if 'AlcoholDrinkers' in encoders else st.selectbox("Alkohol?", ["No", "Yes"])
    input_user['PhysicalHealthDays'] = st.number_input("Hari Fisik Buruk (30 hari terakhir)", 0, 30, 0)
    input_user['MentalHealthDays'] = st.number_input("Hari Mental Buruk (30 hari terakhir)", 0, 30, 0)

for col in feature_names:
    if col not in input_user:
        if col in encoders:
            input_user[col] = encoders[col].classes_[0]
        else:
            input_user[col] = 0.0

st.markdown("---")
selected_model = st.selectbox("Pilih Model AI:", ["Neural Network Base (MLP)", "TabNet (Pretrained)", "Pretrained Style NN"])

if st.button("MULAI ANALISIS SEKARANG"):
    df_input = pd.DataFrame([input_user])[feature_names]
    
    for col, le in encoders.items():
        if col in df_input.columns:
            try:
                df_input[col] = le.transform(df_input[col].astype(str))
            except:
                df_input[col] = 0
    
    X_scaled = scaler.transform(df_input)
    X_tensor = torch.tensor(X_scaled.astype(np.float32))

    with st.spinner('AI sedang menganalisis data Anda...'):
        if selected_model == "Neural Network Base (MLP)":
            with torch.no_grad():
                prob = torch.softmax(mlp_model(X_tensor), dim=1)[0][1].item()
        elif selected_model == "TabNet (Pretrained)":
            prob = tabnet_model.predict_proba(X_scaled)[0][1]
        else:
            with torch.no_grad():
                prob = torch.softmax(pt_style_model(X_tensor), dim=1)[0][1].item()

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.write("### Hasil Analisis Risiko")
    
    res_col1, res_col2 = st.columns([1, 2])
    risk_percent = prob * 100
    
    with res_col1:
        st.metric("Skor Risiko", f"{risk_percent:.1f}%")
    
    with res_col2:
        if risk_percent < 15:
            st.success("🟢 **RISIKO RENDAH**\nJantung Anda terlihat sehat. Pertahankan!")
        elif risk_percent < 30:
            st.warning("🟡 **RISIKO MENENGAH**\nPerhatikan gaya hidup Anda mulai sekarang.")
        else:
            st.error("🔴 **RISIKO TINGGI**\nSangat disarankan konsultasi dengan Dokter Spesialis Jantung.")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("\n\n*Disclaimer: Aplikasi ini hanyalah tugas praktikum, bukan alat diagnosis medis resmi.*")