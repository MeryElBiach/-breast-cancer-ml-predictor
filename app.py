# ==================== app.py ====================
import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="BreastGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS INJECTION (via components — bypasses Streamlit sanitiser) ==========
components.html("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Fraunces:wght@300;700&display=swap" rel="stylesheet">
<style>
  /* tell the iframe to be invisible */
  body { margin:0; padding:0; overflow:hidden; }
</style>
<script>
(function injectStyles() {
  const css = `
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Fraunces:wght@300;700&display=swap');

    :root {
      --navy:#0d1b2a; --teal:#0ea5e9; --cyan:#06b6d4;
      --rose:#f43f5e; --slate:#1e293b; --muted:#64748b;
      --border:#e2e8f0; --bg:#f0f5fa; --r:14px;
      --sh:0 2px 12px rgba(13,27,42,.07);
      --sh2:0 8px 30px rgba(13,27,42,.13);
    }

    /* ── BASE ── */
    .stApp { background:var(--bg)!important; font-family:'DM Sans',sans-serif; }
    .block-container { padding:2rem 2.5rem 4rem!important; max-width:1080px!important; }
    header[data-testid="stHeader"] { background:transparent!important; }

    /* ── SIDEBAR BACKGROUND ── */
    [data-testid="stSidebar"] > div:first-child { background:#071422!important; }
    [data-testid="stSidebar"] { background:#071422!important; }

    /* ── SIDEBAR NATIVE TEXT ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label { color:#94a3b8!important; font-size:0.86rem; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color:#e0f2fe!important; }
    [data-testid="stSidebar"] strong { color:#67e8f9!important; }

    /* ── SLIDERS — teal ── */
    [data-testid="stSlider"] [role="slider"] {
      background:var(--teal)!important;
      border-color:var(--teal)!important;
      box-shadow:0 0 0 4px rgba(14,165,233,.18)!important;
    }
    [data-testid="stSlider"] [data-testid="stThumbValue"] {
      color:var(--teal)!important; font-weight:700!important;
    }
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
      color:var(--muted)!important; font-size:.78rem!important;
    }

    /* ── BUTTON ── */
    .stButton > button {
      background:linear-gradient(135deg,var(--teal),var(--cyan))!important;
      color:white!important; border:none!important; border-radius:50px!important;
      font-family:'DM Sans',sans-serif!important; font-weight:600!important;
      font-size:1rem!important; padding:.75rem 2.5rem!important;
      transition:all .25s ease!important;
      box-shadow:0 4px 16px rgba(14,165,233,.35)!important;
    }
    .stButton > button:hover {
      transform:translateY(-2px)!important;
      box-shadow:0 8px 28px rgba(14,165,233,.5)!important;
    }

    /* ── HERO ── */
    .bg-hero {
      background:linear-gradient(135deg,#071422 0%,#0f2744 55%,#083040 100%);
      border-radius:22px; padding:3rem 3.5rem; margin-bottom:2rem;
      position:relative; overflow:hidden;
      animation:fadeDown .7s ease both;
    }
    .bg-hero::before {
      content:''; position:absolute; inset:0; pointer-events:none;
      background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%230ea5e9' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/svg%3E");
    }
    .hero-badge {
      display:inline-flex; align-items:center; gap:6px;
      background:rgba(14,165,233,.15); border:1px solid rgba(14,165,233,.3);
      color:#7dd3fc; border-radius:50px; padding:4px 14px;
      font-size:.75rem; font-weight:700; letter-spacing:.9px;
      text-transform:uppercase; margin-bottom:1rem;
    }
    .bg-hero h1 {
      font-family:'Fraunces',Georgia,serif; font-size:clamp(2rem,4vw,3rem);
      font-weight:700; color:white; line-height:1.15; margin-bottom:.75rem;
    }
    .bg-hero h1 span { color:#67e8f9; }
    .bg-hero p { color:#94a3b8; font-size:1.05rem; max-width:540px; line-height:1.65; }
    .hero-deco {
      position:absolute; right:3.5rem; top:50%; transform:translateY(-50%);
      font-size:7rem; opacity:.06; pointer-events:none;
    }

    /* ── DISCLAIMER ── */
    .disclaimer {
      background:linear-gradient(135deg,#fff7ed,#fef3c7);
      border:1px solid #fde68a; border-left:4px solid #f59e0b;
      border-radius:var(--r); padding:1rem 1.25rem; margin-bottom:2rem;
      display:flex; gap:12px; align-items:flex-start;
      animation:fadeDown .8s ease .1s both;
    }
    .disclaimer p { color:#92400e; font-size:.88rem; line-height:1.55; }
    .disclaimer strong { color:#78350f; }

    /* ── SECTION LABELS ── */
    .sec-title {
      font-family:'Fraunces',serif; font-size:1.45rem; font-weight:700;
      color:var(--navy); margin-bottom:.25rem;
    }
    .sec-sub { color:var(--muted); font-size:.88rem; margin-bottom:1.5rem; }

    /* ── FEATURE CARDS — fixed equal height ── */
    .feat-card {
      background:white; border-radius:var(--r); border:1px solid var(--border);
      padding:1rem 1.25rem; height:88px;
      display:flex; flex-direction:column; justify-content:center;
      box-shadow:var(--sh); overflow:hidden;
      transition:box-shadow .2s, border-color .2s;
      animation:fadeUp .5s ease both;
    }
    .feat-card:hover { box-shadow:var(--sh2); border-color:#bae6fd; }
    .feat-name {
      font-weight:700; color:var(--navy); font-size:.9rem;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }
    .feat-desc {
      color:var(--muted); font-size:.77rem; margin-top:3px; line-height:1.35;
      display:-webkit-box; -webkit-line-clamp:2;
      -webkit-box-orient:vertical; overflow:hidden;
    }

    /* ── RESULT CARDS ── */
    .result-card { border-radius:20px; padding:2rem 2.5rem; margin-bottom:1.25rem; animation:fadeUp .4s ease both; }
    .result-card.benign { background:linear-gradient(135deg,#ecfdf5,#d1fae5); border:1.5px solid #6ee7b7; }
    .result-card.malign { background:linear-gradient(135deg,#fff1f2,#ffe4e6); border:1.5px solid #fda4af; }
    .res-tag {
      display:inline-flex; align-items:center; gap:5px;
      font-size:.74rem; font-weight:800; letter-spacing:.9px; text-transform:uppercase;
      border-radius:50px; padding:4px 14px; margin-bottom:.7rem; color:white;
    }
    .res-tag.benign { background:#10b981; }
    .res-tag.malign { background:#f43f5e; }
    .res-title { font-family:'Fraunces',serif; font-size:1.75rem; font-weight:700; margin-bottom:.4rem; }
    .res-title.benign { color:#065f46; }
    .res-title.malign { color:#9f1239; }
    .res-body { font-size:.93rem; line-height:1.6; }
    .res-body.benign { color:#047857; }
    .res-body.malign { color:#be123c; }

    /* ── RECO ── */
    .reco-card { border-radius:var(--r); padding:1.25rem 1.5rem; margin-top:.75rem; display:flex; gap:1rem; align-items:flex-start; }
    .reco-card.benign { background:#f0fdf4; border:1px solid #bbf7d0; }
    .reco-card.malign { background:#fff1f2; border:1px solid #fecdd3; }
    .reco-icon { font-size:1.7rem; flex-shrink:0; margin-top:3px; }
    .reco-title { font-weight:700; font-size:.93rem; margin-bottom:.4rem; }
    .reco-title.benign { color:#065f46; }
    .reco-title.malign { color:#9f1239; }
    .reco-list { list-style:none; padding:0; }
    .reco-list li { font-size:.84rem; color:var(--slate); padding:2px 0; display:flex; gap:7px; }
    .reco-list li::before { content:'→'; color:var(--teal); font-weight:800; flex-shrink:0; }

    /* ── PROBABILITY ── */
    .prob-card { background:white; border-radius:var(--r); border:1px solid var(--border); padding:1.5rem; box-shadow:var(--sh); }
    .prob-head { font-family:'Fraunces',serif; font-weight:700; font-size:1rem; color:var(--navy); margin-bottom:1.25rem; }
    .prob-lbl { display:flex; justify-content:space-between; font-size:.86rem; font-weight:600; color:var(--slate); margin-bottom:.4rem; }
    .prob-track { background:#f1f5f9; border-radius:50px; height:11px; overflow:hidden; margin-bottom:1.1rem; }
    .prob-fill { height:100%; border-radius:50px; transition:width 1.2s cubic-bezier(.23,1,.32,1); }
    .prob-fill.benign { background:linear-gradient(90deg,#0ea5e9,#06b6d4); }
    .prob-fill.malign { background:linear-gradient(90deg,#f43f5e,#fb7185); }
    .conf-badge { background:#f8fafc; border-radius:11px; padding:.9rem 1rem; text-align:center; margin-top:.5rem; }
    .conf-label { font-size:.7rem; color:var(--muted); font-weight:700; letter-spacing:.7px; text-transform:uppercase; margin-bottom:3px; }
    .conf-val   { font-size:1.5rem; font-weight:800; }
    .conf-pct   { font-size:.75rem; color:#94a3b8; margin-top:2px; }

    /* ── FOOTER ── */
    .app-footer {
      text-align:center; padding:2rem 0 .5rem; color:#94a3b8; font-size:.8rem;
      line-height:1.8; border-top:1px solid var(--border); margin-top:3rem;
    }
    .app-footer a { color:var(--teal); text-decoration:none; }

    /* ── ANIMATIONS ── */
    @keyframes fadeDown { from{opacity:0;transform:translateY(-18px)} to{opacity:1;transform:translateY(0)} }
    @keyframes fadeUp   { from{opacity:0;transform:translateY(14px)}  to{opacity:1;transform:translateY(0)} }
  `;
  const tag = document.createElement('style');
  tag.textContent = css;
  // inject into parent window (Streamlit's main document)
  try {
    window.parent.document.head.appendChild(tag);
  } catch(e) {
    document.head.appendChild(tag);
  }
})();
</script>
""", height=0, scrolling=False)


# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    model         = joblib.load('best_model.pkl')
    imputer       = joblib.load('imputer.pkl')
    feature_names = joblib.load('feature_names.pkl')
    model_info    = joblib.load('model_info.pkl')
    scaler        = joblib.load('scaler.pkl') if model_info['needs_scaling'] else None
    return model, imputer, scaler, feature_names, model_info

try:
    model, imputer, scaler, feature_names, model_info = load_models()
    ok = True
except Exception as e:
    st.error(f"❌ Cannot load model files: {e}")
    ok = False


# ========== SIDEBAR (100% native Streamlit — no custom HTML) ==========
with st.sidebar:
    # Logo row via native markdown — styled by injected CSS
    st.markdown("""
<div style="display:flex;align-items:center;gap:12px;
            padding-bottom:1.4rem;margin-bottom:1.6rem;
            border-bottom:1px solid rgba(6,182,212,0.2)">
  <div style="width:46px;height:46px;flex-shrink:0;border-radius:13px;
              background:linear-gradient(135deg,#0ea5e9,#06b6d4);
              display:flex;align-items:center;justify-content:center;
              font-size:1.4rem;box-shadow:0 4px 18px rgba(6,182,212,0.5)">🩺</div>
  <div>
    <div style="font-size:1.15rem;font-weight:800;color:#e0f2fe;line-height:1.1;">
      Breast<span style="background:linear-gradient(90deg,#38bdf8,#22d3ee);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Guard AI</span>
    </div>
    <div style="font-size:0.68rem;color:#3a7a94;letter-spacing:0.8px;
                text-transform:uppercase;margin-top:3px;">Tumor Classification Tool</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.68rem;font-weight:800;letter-spacing:1.4px;'
        'text-transform:uppercase;color:#38bdf8;margin-bottom:0.9rem;">How it works</p>',
        unsafe_allow_html=True
    )

    steps = [
        ("Enter measurements", "Adjust the 9 sliders using values from your lab report or physician."),
        ("Run the analysis",   "Click the button — the AI model evaluates the pattern instantly."),
        ("Review results",     "See a clear prediction, confidence score, and recommended actions."),
        ("See a specialist",   "Always confirm with a qualified oncologist before any medical decision."),
    ]
    for n, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
<div style="display:flex;gap:12px;align-items:flex-start;
            background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.15);
            border-radius:11px;padding:0.85rem 1rem;margin-bottom:0.75rem;
            transition:background .2s;">
  <div style="width:26px;height:26px;border-radius:7px;flex-shrink:0;
              background:linear-gradient(135deg,#0ea5e9,#06b6d4);
              color:white;font-size:0.78rem;font-weight:800;
              display:flex;align-items:center;justify-content:center;
              box-shadow:0 2px 8px rgba(14,165,233,0.4);">{n}</div>
  <div>
    <div style="color:#dbeafe;font-size:0.86rem;font-weight:700;margin-bottom:2px;">{title}</div>
    <div style="color:#5eacc8;font-size:0.77rem;line-height:1.5;">{desc}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top:1.5rem;padding-top:1.25rem;
            border-top:1px solid rgba(6,182,212,0.14);
            color:#2f6070;font-size:0.75rem;line-height:1.8;text-align:center;">
  <span style="color:#4db8d4;font-weight:700;">🔒 No data is stored or shared.</span><br>
  For educational &amp; demonstration use only.
</div>
""", unsafe_allow_html=True)


# ========== HERO ==========
st.markdown("""
<div class="bg-hero">
  <div class="hero-badge">🧬 AI-Powered Screening Tool</div>
  <h1>Breast Tumor <span>Classification</span></h1>
  <p>Enter nine cell-level measurements below. Our machine learning model predicts whether the tumor pattern is benign or malignant — clearly and instantly.</p>
  <div class="hero-deco">🩺</div>
</div>
<div class="disclaimer">
  <span style="font-size:1.1rem;flex-shrink:0;margin-top:1px">⚠️</span>
  <p><strong>Medical Disclaimer:</strong> This tool is for <strong>educational and demonstration purposes only.</strong>
  It does not constitute a medical diagnosis. Always consult a qualified healthcare professional.</p>
</div>
""", unsafe_allow_html=True)


# ========== FEATURE INPUTS ==========
META = {
    'ClumpThickness':           ('Clump Thickness',         'How thickly the cells are grouped together'),
    'UniformityCellSize':       ('Cell Size Uniformity',    'Consistency of cell sizes across the sample'),
    'UniformityCellShape':      ('Cell Shape Uniformity',   'Regularity of the cell shape observed'),
    'MarginalAdhesion':         ('Marginal Adhesion',       'How strongly cells stick together at the edges'),
    'SingleEpithelialCellSize': ('Epithelial Cell Size',    'Size of individual epithelial cells'),
    'BareNuclei':               ('Bare Nuclei',             'Nuclei without surrounding cytoplasm'),
    'BlandChromatin':           ('Bland Chromatin',         'Texture uniformity of nuclear chromatin'),
    'NormalNucleoli':           ('Normal Nucleoli',         'Appearance and size of the nucleoli'),
    'Mitoses':                  ('Mitotic Activity',        'Rate of cell division in the sample'),
}

if ok:
    st.markdown('<div class="sec-title">Cell Feature Measurements</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">All values range from 1 (normal) to 10 (most abnormal), as defined in the Wisconsin Breast Cancer Dataset.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    input_data = {}

    for i, feat in enumerate(feature_names):
        label, desc = META.get(feat, (feat, ''))
        col = [col1, col2, col3][i % 3]
        with col:
            st.markdown(f"""
<div class="feat-card" style="animation-delay:{i*0.05}s">
  <div class="feat-name">{label}</div>
  <div class="feat-desc">{desc}</div>
</div>""", unsafe_allow_html=True)
            input_data[feat] = st.slider(
                label, min_value=1, max_value=10, value=5,
                key=feat, label_visibility="collapsed"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1.5, 2, 1.5])
    with mid:
        go = st.button("🔍  Analyse Tumor Features", type="primary", use_container_width=True)

    # ========== PREDICTION ==========
    if go:
        with st.spinner("Running AI analysis…"):
            df_in    = pd.DataFrame([input_data])
            df_imp   = pd.DataFrame(imputer.transform(df_in), columns=feature_names)
            df_final = scaler.transform(df_imp) if scaler else df_imp
            pred     = model.predict(df_final)[0]
            proba    = model.predict_proba(df_final)[0]

        pb = proba[0] * 100
        pm = proba[1] * 100
        is_benign = pred == 0

        st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:2.5rem 0 1.5rem">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([3, 2], gap="large")

        with left:
            if is_benign:
                st.markdown("""
<div class="result-card benign">
  <div class="res-tag benign">✓ Result</div>
  <div class="res-title benign">Benign Tumor</div>
  <div class="res-body benign">
    The AI model found no patterns associated with malignancy.
    The tumor appears <strong>non-cancerous</strong>.
    Please confirm this result with your physician.
  </div>
</div>
<div class="reco-card benign">
  <div class="reco-icon">📋</div>
  <div>
    <div class="reco-title benign">Recommended Next Steps</div>
    <ul class="reco-list">
      <li>Schedule a follow-up with your GP for clinical confirmation</li>
      <li>Continue routine annual breast examinations</li>
      <li>Monitor any physical changes and report them promptly</li>
      <li>Maintain a healthy lifestyle to reduce future risk</li>
    </ul>
  </div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="result-card malign">
  <div class="res-tag malign">⚠ Alert</div>
  <div class="res-title malign">Malignant Tumor</div>
  <div class="res-body malign">
    The AI model detected patterns consistent with malignancy.
    The tumor may be <strong>cancerous</strong>.
    Please seek specialist consultation <strong>immediately</strong>.
  </div>
</div>
<div class="reco-card malign">
  <div class="reco-icon">🚨</div>
  <div>
    <div class="reco-title malign">Urgent Actions Required</div>
    <ul class="reco-list">
      <li>Contact an oncologist or breast specialist without delay</li>
      <li>Request confirmatory imaging (MRI, ultrasound) and biopsy</li>
      <li>Do not delay — early diagnosis greatly improves outcomes</li>
      <li>Bring all available medical records to your appointment</li>
    </ul>
  </div>
</div>""", unsafe_allow_html=True)

        with right:
            dom    = max(pb, pm)
            clabel = "High" if dom >= 80 else ("Moderate" if dom >= 60 else "Low")
            ccolor = "#0ea5e9" if dom >= 80 else ("#f59e0b" if dom >= 60 else "#f43f5e")
            st.markdown(f"""
<div class="prob-card">
  <div class="prob-head">Confidence Levels</div>
  <div class="prob-lbl">
    <span>🔵 Benign</span>
    <span style="color:#0ea5e9">{pb:.1f}%</span>
  </div>
  <div class="prob-track"><div class="prob-fill benign" style="width:{pb:.1f}%"></div></div>
  <div class="prob-lbl">
    <span>🔴 Malignant</span>
    <span style="color:#f43f5e">{pm:.1f}%</span>
  </div>
  <div class="prob-track"><div class="prob-fill malign" style="width:{pm:.1f}%"></div></div>
  <div class="conf-badge">
    <div class="conf-label">Model Confidence</div>
    <div class="conf-val" style="color:{ccolor}">{clabel}</div>
    <div class="conf-pct">{dom:.1f}% certainty</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊  View all entered values"):
            df_show = pd.DataFrame({
                'Feature':      [META.get(f, (f,))[0] for f in feature_names],
                'Value (1–10)': list(input_data.values())
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True)

# ========== FOOTER ==========
st.markdown("""
<div class="app-footer">
  <strong>BreastGuard AI</strong> — Built with Streamlit &amp; Scikit-learn &nbsp;·&nbsp;
  Dataset: <a href="https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)" target="_blank">Wisconsin Breast Cancer Database (UCI)</a>
  &nbsp;·&nbsp; For educational use only
</div>
""", unsafe_allow_html=True)