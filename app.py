import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import io
import base64

st.set_page_config(page_title="TerraVision", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@300;400;500&display=swap');

*,
*::before,
*::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html,
body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: #1a1a1a !important;
    color: #ffffff;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"],
footer,
#MainMenu,
[data-testid="stSidebar"] {
    display: none !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

section[data-testid="stMain"] > div {
    padding: 0 !important;
}

.navbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1rem 2.5rem;
    border-bottom: 1px solid #2e2e2e;
}

.logo-box {
    width: 36px;
    height: 36px;
    border: 1.5px solid #00ffa0;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    background: rgba(0,255,160,0.07);
    flex-shrink: 0;
}

.logo-text,
.logo-text:link,
.logo-text:visited,
.logo-text:active {
    font-family: 'Orbitron', monospace;
    font-size: 30px;
    font-weight: 700;
    color: #ffffff !important;
    cursor: pointer;
    text-decoration: none !important;
}

.logo-text:hover { color: #00ffa0 !important; }
.logo-text em { color: #00ffa0; font-style: normal; }

.home { text-align: center; padding: 5rem 2rem 3rem; }

.home h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 900;
    color: #fff;
    margin-bottom: 1.2rem;
}

.home h1 em { color: #00ffa0; font-style: normal; }

.home p {
    font-size: 1.15rem;
    color: #fff;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
    font-weight: 300;
}

[data-testid="stFileUploader"] { background: transparent !important; }

[data-testid="stFileUploader"] section {
    background: #222 !important;
    border: 1.5px dashed #383838 !important;
    border-radius: 14px !important;
    padding: 2.5rem !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"] section:hover { border-color: #00ffa0 !important; }

[data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: #777 !important;
    font-size: 1rem !important;
}

[data-testid="stFileUploader"] button {
    background: #2a2a2a !important;
    border: 1px solid #444 !important;
    color: #fff !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
}

[data-testid="stFileUploader"] button:hover { border-color: #00ffa0 !important; }

.results-page { padding: 1.5rem 2rem; }

.card {
    background: #222;
    border: 1px solid #2e2e2e;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

.card-label {
    font-family: 'Orbitron', monospace !important;
    font-size: 15px;
    letter-spacing: 3px;
    color: #bbbbbb;
    text-transform: uppercase;
    padding-bottom: 0.6rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #2e2e2e;
}

.result-center { text-align: center; padding: 0.3rem 0 0.6rem; }
.result-emoji { font-size: 3rem; display: block; margin-bottom: 0.3rem; }
.result-label { font-size: 0.9rem; letter-spacing: 2px; color: #ffffff; margin-bottom: 0.2rem; }

.result-name {
    font-family: 'Orbitron', monospace !important;
    font-size: 2rem;
    font-weight: 900;
    color: #fff;
    margin-bottom: 0.8rem;
}

.bar-row { display: flex; justify-content: space-between; margin-bottom: 5px; }
.bar-row span { font-size: 0.95rem; color: #ffffff; }
.bar-row strong { font-family: 'Orbitron', monospace !important; font-size: 1rem; font-weight: 700; }
.bar-track { height: 6px; background: #2e2e2e; border-radius: 4px; overflow: hidden; margin-bottom: 8px; }
.bar-fill { height: 100%; border-radius: 4px; }

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

.high strong { color: #00ffa0; }
.high .bar-fill { background: linear-gradient(90deg, #00c878, #00ffa0); }
.high .badge { background: rgba(0,255,160,0.1); color: #00ffa0; border: 1px solid rgba(0,255,160,0.3); }

.med strong { color: #f0b840; }
.med .bar-fill { background: linear-gradient(90deg, #b08010, #f0b840); }
.med .badge { background: rgba(240,184,64,0.1); color: #f0b840; border: 1px solid rgba(240,184,64,0.3); }

.low strong { color: #ff5060; }
.low .bar-fill { background: linear-gradient(90deg, #c03040, #ff5060); }
.low .badge { background: rgba(255,80,96,0.1); color: #ff5060; border: 1px solid rgba(255,80,96,0.3); }

.pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #2a2a2a;
}

.pred-row:last-child { border-bottom: none; }
.pred-num { font-family: 'Orbitron', monospace !important; font-size: 0.65rem; color: #aaaaaa; width: 20px; }
.pred-icon { font-size: 1.2rem; width: 24px; text-align: center; }
.pred-name { flex: 1; color: #ffffff; font-weight: 500; font-size: 1.05rem; }
.pred-name.top { color: #ffffff; }
.pred-bar { flex: 1; height: 3px; background: #2e2e2e; border-radius: 2px; overflow: hidden; }
.pred-bar-inner { height: 100%; background: #383838; border-radius: 2px; }
.pred-bar-inner.top { background: #00ffa0; }
.pred-pct { font-family: 'Orbitron', monospace !important; font-size: 0.9rem; color: #ffffff; min-width: 44px; text-align: right; }
.pred-pct.top { color: #00ffa0; }

.img-tags { display: flex; gap: 8px; margin-top: 8px; }

.img-tag {
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 6px;
    background: #282828;
    border: 1px solid #333;
    color: #ccc;
}

button[title="View fullscreen"] { display: none !important; }

div[data-testid="stButton"] > button {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    color: #ffffff !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    width: 100%;
}

div[data-testid="stButton"] > button:hover {
    border-color: #00ffa0 !important;
    color: #00ffa0 !important;
}

.stSpinner > div { border-top-color: #00ffa0 !important; }
</style>
""", unsafe_allow_html=True)


# ResNet18
@st.cache_resource
def load_model():
    if not os.path.exists("model.pth"):
        return None, None
    ck          = torch.load("model.pth", map_location="cpu")
    class_names = ck["class_names"]
    num_classes  = ck["num_classes"]

    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, class_names

def run_predict(image, model, class_names):
    t = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        p = torch.softmax(model(t(image).unsqueeze(0)), dim=1)
        conf, idx = torch.max(p, dim=1)
    return (class_names[idx.item()], conf.item(),
            {class_names[i]: p[0][i].item() for i in range(len(class_names))})

CLASS_META = {
    "AnnualCrop":           ("🌾", "Annual Crop"),
    "Forest":               ("🌲", "Forest"),
    "HerbaceousVegetation": ("🌿", "Herbaceous"),
    "Highway":              ("🛣️",  "Highway"),
    "Industrial":           ("🏭", "Industrial"),
    "Pasture":              ("🐄", "Pasture"),
    "PermanentCrop":        ("🍇", "Perm. Crop"),
    "Residential":          ("🏘️",  "Residential"),
    "River":                ("🌊", "River"),
    "SeaLake":              ("🏖️",  "Sea / Lake"),
}

model, class_names = load_model()
if model is None:
    st.error("model.pth not found — run `python main.py` first.")
    st.stop()

if "page" not in st.session_state:
    st.session_state.page = "home"
if "image" not in st.session_state:
    st.session_state.image = None

params = st.query_params
if params.get("go") == "home":
    st.session_state.page  = "home"
    st.session_state.image = None
    st.query_params.clear()
    st.rerun()

st.markdown("""
<div class="navbar">
  <div class="logo-box">🛰️</div>
  <a class="logo-text" href="?go=home">Terra<em>Vision</em></a>
</div>
""", unsafe_allow_html=True)


# HOME PAGE
if st.session_state.page == "home":
    st.markdown("""
    <div class="home">
      <h1>Terra<em>Vision</em></h1>
      <p>Upload a satellite image and TerraVision will identify the terrain type with a confidence score.</p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "tif"], label_visibility="collapsed")
        if uploaded:
            st.session_state.image = uploaded.read()
            st.session_state.page  = "results"
            st.rerun()


# RESULTS PAGE
else:
    raw   = st.session_state.image
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    with st.spinner("Analyzing terrain..."):
        pred_class, confidence, all_probs = run_predict(image, model, class_names)

    emoji, label = CLASS_META.get(pred_class, ("🗺️", pred_class))
    pct = confidence * 100
    cc  = "high" if confidence >= 0.85 else "med" if confidence >= 0.60 else "low"
    cw  = {"high": "High Confidence", "med": "Medium Confidence", "low": "Low Confidence"}[cc]
    sp  = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    mx  = sp[0][1]

    st.markdown('<div class="results-page">', unsafe_allow_html=True)
    col_img, col_res = st.columns([1, 1.4])

    with col_img:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        st.markdown(f"""
<div class="card">
<div class="card-label">📡 &nbsp; Input Image</div>
<img src="data:image/png;base64,{img_str}" style="width:100%; border-radius:10px; margin-bottom:10px;">
<div class="img-tags">
  <span class="img-tag">RGB SATELLITE</span>
  <span class="img-tag">{image.size[0]} × {image.size[1]} px</span>
</div>
</div>
""", unsafe_allow_html=True)

        if st.button("← Classify another image", use_container_width=True):
            st.session_state.page  = "home"
            st.session_state.image = None
            st.rerun()

    with col_res:
        st.markdown(f"""
        <div class="card {cc}">
          <div class="card-label">🎯 &nbsp; Classification Result</div>
          <div class="result-center">
            <span class="result-emoji">{emoji}</span>
            <div class="result-label">Predicted Land Type</div>
            <div class="result-name">{label}</div>
            <div class="bar-row">
              <span>Confidence Score</span>
              <strong>{pct:.1f}%</strong>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
            <span class="badge">{cw}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        rows_html = '<div class="card"><div class="card-label">📊 &nbsp; All Predictions</div>'
        for i, (cls, prob) in enumerate(sp):
            e, l = CLASS_META.get(cls, ("🗺️", cls))
            top  = "top" if i == 0 else ""
            bw   = (prob / mx) * 100 if mx > 0 else 0
            rows_html += f"""
            <div class="pred-row">
              <div class="pred-num">0{i+1}</div>
              <div class="pred-icon">{e}</div>
              <div class="pred-name {top}">{l}</div>
              <div class="pred-bar"><div class="pred-bar-inner {top}" style="width:{bw:.0f}%"></div></div>
              <div class="pred-pct {top}">{prob*100:.1f}%</div>
            </div>"""
        rows_html += '</div>'
        st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)