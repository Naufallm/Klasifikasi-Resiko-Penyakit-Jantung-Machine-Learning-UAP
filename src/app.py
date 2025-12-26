import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

st.set_page_config(page_title="HeartHealth AI Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0a192f; font-family: 'Segoe UI', sans-serif; color: #e6f1ff; }
    [data-testid="stSidebar"] { background-color: #112240; border-right: 1px solid #233554; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background: linear-gradient(90deg, #64ffda 0%, #48d1cc 100%); color: #0a192f; font-weight: bold; border: none; }
    .result-card { padding: 30px; border-radius: 15px; background-color: #112240; border: 1px solid #233554; margin-top: 20px; }
    .stSelectbox label, .stNumberInput label, .stSlider label { color: #8892b0 !important; font-size: 16px; }
    h1, h2, h3 { color: #64ffda !important; }
    </style>
    """, unsafe_allow_html=True)

class BaseMLP(nn.Module):
    def __init__(self, input_dim):
        super(BaseMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

class FTTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=32, n_heads=4, n_layers=2):
        super(FTTransformer, self).__init__()
        self.tokenizer = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(input_dim)])
        # Ubah dim_feedforward dari 64 menjadi 128 sesuai dengan hasil training Anda
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=128, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Linear(embed_dim, 2)
    def forward(self, x):
        tokens = [self.tokenizer[i](x[:, i].unsqueeze(-1)) for i in range(x.shape[1])]
        x = torch.stack(tokens, dim=1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0])

@st.cache_resource
def load_assets():
    scaler = joblib.load('src/models/scaler.pkl')
    feature_names = joblib.load('src/models/feature_names.pkl')
    encoders = joblib.load('src/models/label_encoders.pkl')
    mlp = BaseMLP(len(feature_names))
    mlp.load_state_dict(torch.load('src/models/mlp_model.pth', map_location='cpu'))
    mlp.eval()
    ft_model = FTTransformer(len(feature_names))
    ft_model.load_state_dict(torch.load('src/models/ft_transformer_model.pth', map_location='cpu'))
    ft_model.eval()
    tabnet = TabNetClassifier()
    tabnet.load_model('src/models/tabnet_model.zip')
    return scaler, feature_names, encoders, mlp, ft_model, tabnet

scaler, feature_names, encoders, mlp_model, ft_model, tabnet_model = load_assets()

with st.sidebar:
    st.title("🛡️ Analysis Control")
    model_choice = st.radio("Pilih Arsitektur Model:", ["Model 1: Base MLP", "Model 2: TabNet Pretrained", "Model 3: FT-Transformer"])
    st.markdown("---")
    st.caption("Aplikasi Prediksi Risiko Jantung")

st.title("❤️ Heart Health AI Predictor")
st.markdown("Analisis risiko berdasarkan data klinis dan profil kesehatan Anda.")

input_user = {}
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        input_user['GeneralHealth'] = st.selectbox("Kesehatan Umum", [c for c in encoders['GeneralHealth'].classes_ if str(c) != 'nan'])
        input_user['Sex'] = st.selectbox("Jenis Kelamin", [c for c in encoders['Sex'].classes_ if str(c) != 'nan'])
        input_user['AgeCategory'] = st.selectbox("Kategori Usia", [c for c in encoders['AgeCategory'].classes_ if str(c) != 'nan'])
        input_user['SmokerStatus'] = st.selectbox("Status Merokok", [c for c in encoders['SmokerStatus'].classes_ if str(c) != 'nan'])
        input_user['PhysicalActivities'] = st.selectbox("Aktif Berolahraga?", [c for c in encoders['PhysicalActivities'].classes_ if str(c) != 'nan'])
    with col2:
        input_user['HadStroke'] = st.selectbox("Pernah Mengalami Stroke?", [c for c in encoders['HadStroke'].classes_ if str(c) != 'nan'])
        input_user['HadDiabetes'] = st.selectbox("Menderita Diabetes?", [c for c in encoders['HadDiabetes'].classes_ if str(c) != 'nan'])
        input_user['AlcoholDrinkers'] = st.selectbox("Peminum Alkohol?", [c for c in encoders['AlcoholDrinkers'].classes_ if str(c) != 'nan'])
        input_user['PhysicalHealthDays'] = st.slider("Hari Fisik Buruk (30 hari)", 0, 30, 0)
        w = st.number_input("Berat (kg)", 20.0, 250.0, 70.0)
        h = st.number_input("Tinggi (m)", 0.9, 2.5, 1.7)
        input_user['BMI'], input_user['HeightInMeters'], input_user['WeightInKilograms'] = w/(h**2), h, w

for col in feature_names:
    if col not in input_user:
        if col in encoders: input_user[col] = [c for c in encoders[col].classes_ if str(c) != 'nan'][0]
        else: input_user[col] = 7.0 if col == 'SleepHours' else 0.0

if st.button("MULAI ANALISIS KESEHATAN"):
    df_input = pd.DataFrame([input_user])[feature_names]
    for col, le in encoders.items():
        if col in df_input.columns: df_input[col] = le.transform(df_input[col].astype(str))
    X_scaled = scaler.transform(df_input)
    X_tensor = torch.tensor(X_scaled.astype(np.float32))
    with st.spinner('Analisis...'):
        if "Model 1" in model_choice: prob = torch.softmax(mlp_model(X_tensor), 1)[0][1].item()
        elif "Model 2" in model_choice: prob = tabnet_model.predict_proba(X_scaled)[0][1]
        else: prob = torch.softmax(ft_model(X_tensor), 1)[0][1].item()
    res_col1, res_col2 = st.columns([1, 2])
    risk = prob * 100
    with res_col1: st.metric("Probability Score", f"{risk:.1f}%")
    with res_col2:
        if risk < 35: st.success("### ✅ RISIKO RENDAH")
        elif risk < 70: st.warning("### ⚠️ RISIKO MODERAT")
        else: st.error("### 🚨 RISIKO TINGGI")
    st.markdown('</div>', unsafe_allow_html=True)