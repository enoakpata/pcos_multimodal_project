import streamlit as st
import tempfile
import os
from infer import PCOSInference

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Diagnostic Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e8e6e1;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a1f35 0%, #0f1117 60%);
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #f0ede8;
}

.app-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #2a2d3a;
    margin-bottom: 2.5rem;
}
.app-header h1 {
    font-size: 2.6rem;
    letter-spacing: -0.5px;
    margin: 0;
    background: linear-gradient(135deg, #c9b8f5, #7eb8f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header p {
    color: #6b7280;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

.section-card {
    background: #161820;
    border: 1px solid #252836;
    border-radius: 14px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #c9b8f5;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.result-positive {
    background: linear-gradient(135deg, #2d1b1b, #1f1010);
    border: 1px solid #7f2d2d;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #1b2d1b, #101f10);
    border: 1px solid #2d7f2d;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0.5rem 0;
}
.result-confidence {
    font-size: 0.9rem;
    color: #9ca3af;
    margin-top: 0.3rem;
}

[data-testid="stFileUploader"] {
    background: #1a1d2e !important;
    border: 1px dashed #3a3d50 !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #1a1d2e !important;
    border-color: #2e3147 !important;
    color: #e8e6e1 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c6fcd, #5b8dee);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

.disclaimer {
    background: #13151f;
    border-left: 3px solid #4b5563;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD INFERENCE ENGINE
# ─────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return PCOSInference(
        cnn_path="cnn_final.pt",
        xgb_path="xgb_final.json",
        meta_path="pcos_metalearner.joblib"
    )


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🔬 PCOS Diagnostic Assistant</h1>
    <p>Multimodal AI — Ultrasound + Clinical Biomarkers</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# ── LEFT: Image Upload ──────────────────────
with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Ultrasound Image</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload ovarian ultrasound",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded ultrasound", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Clinical Data ─────────────────────
with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Clinical Data</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        follicle_r  = st.number_input("Follicle No. (Right)", min_value=0, max_value=50, value=0, step=1)
        follicle_l  = st.number_input("Follicle No. (Left)",  min_value=0, max_value=50, value=0, step=1)
        amh         = st.number_input("AMH (ng/mL)", min_value=0.0, max_value=20.0, value=0.0, step=0.1, format="%.2f")
        cycle_input = st.selectbox("Cycle", ["Regular", "Irregular"])
    with c2:
        weight_gain_input    = st.selectbox("Weight Gain",    ["Yes", "No"])
        hair_growth_input    = st.selectbox("Hair Growth",    ["Yes", "No"])
        skin_darkening_input = st.selectbox("Skin Darkening", ["Yes", "No"])
        pimples_input        = st.selectbox("Pimples",        ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

# ── ENCODE INPUTS ─────────────────────────────
def encode_yn(val):
    return 1 if val == "Yes" else 0

clinical_dict = {
    "Follicle No. (R)":      follicle_r,
    "Follicle No. (L)":      follicle_l,
    "AMH(ng/mL)":            amh,
    "Cycle(R/I)":            1 if cycle_input == "Irregular" else 0,
    "Weight gain(Y/N)":      encode_yn(weight_gain_input),
    "hair growth(Y/N)":      encode_yn(hair_growth_input),
    "Skin darkening (Y/N)":  encode_yn(skin_darkening_input),
    "Pimples(Y/N)":          encode_yn(pimples_input),
}

# ── ANALYSE BUTTON ────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyse = st.button("Run Analysis")

# ── RESULT ───────────────────────────────────
if analyse:
    if uploaded_image is None:
        st.warning("Please upload an ultrasound image before running analysis.")
    else:
        try:
            engine = load_engine()

            # infer.py expects a file path, so save upload to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_image.read())
                tmp_path = tmp.name

            with st.spinner("Analysing..."):
                result = engine.predict_final(tmp_path, clinical_dict)

            os.unlink(tmp_path)  # clean up

            st.markdown("<br>", unsafe_allow_html=True)

            if result["label"] == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div style="font-size:2.5rem"></div>
                    <div class="result-label" style="color:#f87171;">PCOS Positive</div>
                    <div class="result-confidence">Confidence: {result['final_probability']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div style="font-size:2.5rem"></div>
                    <div class="result-label" style="color:#4ade80;">PCOS Negative</div>
                    <div class="result-confidence">Confidence: {(1 - result['final_probability'])*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Sub-model breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            exp1, exp2 = st.columns(2)
            with exp1:
                st.metric("CNN (Ultrasound) Probability", f"{result['p_img']*100:.1f}%")
            with exp2:
                st.metric("XGBoost (Clinical) Probability", f"{result['p_clin']*100:.1f}%")

        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}\n\nMake sure cnn_final.pt, xgb_final.json, and pcos_metalearner.joblib are in the same folder as this app.")
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")

# ── DISCLAIMER ───────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>Clinical Decision Support Only.</strong> This tool is intended to assist clinicians and does not replace professional medical judgment. All results must be reviewed and confirmed by a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)