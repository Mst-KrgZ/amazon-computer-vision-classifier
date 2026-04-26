"""
Product Category Prediction System — Computer Vision
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")

# ─── Page Config ───
st.set_page_config(
    page_title="Product Category Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 0.8rem !important; max-width: 1200px; }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
        border-radius: 16px; padding: 2rem 2.5rem;
        margin-bottom: 1.5rem; text-align: center;
        box-shadow: 0 4px 24px rgba(139,92,246,0.25);
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; color: #f5f3ff; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { color: #c4b5fd; font-size: 0.95rem; margin: 0.4rem 0 0 0; }

    .result-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #2e1065 100%);
        border: 1px solid #6d28d9; border-radius: 14px;
        padding: 1.4rem; text-align: center; color: #f5f3ff;
        box-shadow: 0 4px 16px rgba(109,40,217,0.2);
    }
    .result-card .icon  { font-size: 2.5rem; margin-bottom: 0.4rem; }
    .result-card .name  { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-card .score { font-size: 1.8rem; font-weight: 800; color: #a78bfa; font-family: 'JetBrains Mono', monospace; }
    .result-card .label { font-size: 0.75rem; color: #c4b5fd; text-transform: uppercase; letter-spacing: 1px; }

    .cat-chip {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.35rem 0.75rem; border-radius: 99px;
        font-size: 0.82rem; font-weight: 500; margin: 0.2rem;
    }

    .kpi-card {
        background: #1e1b4b; border: 1px solid #4c1d95;
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    }
    .kpi-card .kpi-v { font-size: 1.8rem; font-weight: 700; color: #a78bfa; font-family: 'JetBrains Mono', monospace; }
    .kpi-card .kpi-l { font-size: 0.75rem; color: #8b7cf6; text-transform: uppercase; letter-spacing: 1px; }

    section[data-testid="stSidebar"] { background: #0f0a1e; border-right: 1px solid #2e1065; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #a78bfa !important; }

    .stTabs [data-baseweb="tab"] { font-size: 0.95rem !important; font-weight: 600 !important; }

    .pill-ok  { background:#14532d20; color:#4ade80; border:1px solid #16a34a; padding:0.25rem 0.7rem; border-radius:99px; font-size:0.8rem; }
    .pill-err { background:#4c0519; color:#f87171; border:1px solid #dc2626; padding:0.25rem 0.7rem; border-radius:99px; font-size:0.8rem; }

    .pipeline-label { font-size:0.72rem; text-align:center; color:#8b7cf6; margin-top:0.3rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───
APP_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = APP_DIR / "mobilenetv2_local.h5"
IMG_SIZE   = (224, 224)

CLASS_NAMES = [
    "BABY_PRODUCTS",
    "BEAUTY_HEALTH",
    "CLOTHING_ACCESSORIES_JEWELLERY",
    "ELECTRONICS",
    "GROCERY",
    "HOBBY_ARTS_STATIONERY",
    "HOME_KITCHEN_TOOLS",
    "PET_SUPPLIES",
    "SPORTS_OUTDOOR",
]

CATEGORY_ICONS = {
    "BABY_PRODUCTS":                    "👶",
    "BEAUTY_HEALTH":                    "💄",
    "CLOTHING_ACCESSORIES_JEWELLERY":   "👗",
    "ELECTRONICS":                      "📱",
    "GROCERY":                          "🛒",
    "HOBBY_ARTS_STATIONERY":            "🎨",
    "HOME_KITCHEN_TOOLS":               "🏠",
    "PET_SUPPLIES":                     "🐾",
    "SPORTS_OUTDOOR":                   "⚽",
}

CATEGORY_COLORS = {
    "BABY_PRODUCTS":                    "#f472b6",
    "BEAUTY_HEALTH":                    "#fb923c",
    "CLOTHING_ACCESSORIES_JEWELLERY":   "#a78bfa",
    "ELECTRONICS":                      "#60a5fa",
    "GROCERY":                          "#34d399",
    "HOBBY_ARTS_STATIONERY":            "#fbbf24",
    "HOME_KITCHEN_TOOLS":               "#f87171",
    "PET_SUPPLIES":                     "#2dd4bf",
    "SPORTS_OUTDOOR":                   "#818cf8",
}

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("**Model Status**")
    if MODEL_PATH.exists():
        st.markdown('<span class="pill-ok">✅ Model loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-err">❌ Model not found</span>', unsafe_allow_html=True)
        st.warning(f"Copy `mobilenetv2_local.h5` to:\n```\n{APP_DIR}\n```")

    st.markdown("---")
    st.markdown("**⚙️ Prediction Settings**")
    top_k = st.slider("Top-K results:", 3, 9, 5)
    confidence_threshold = st.slider("Min confidence threshold (%):", 10, 90, 50)

    st.markdown("---")
    st.markdown("**📦 Categories**")
    for name in CLASS_NAMES:
        icon    = CATEGORY_ICONS.get(name, "📦")
        color   = CATEGORY_COLORS.get(name, "#8b5cf6")
        display = name.replace("_", " ").title()
        st.markdown(
            f'<div style="padding:0.3rem 0.6rem;margin-bottom:3px;border-radius:6px;'
            f'background:{color}18;border-left:3px solid {color};font-size:0.83rem;">'
            f'{icon} {display}</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<p style="color:#4a3f6b;font-size:0.72rem;text-align:center;margin-top:1rem;">'
        'CV Classifier · MobileNetV2 · 9 Categories'
        '</p>',
        unsafe_allow_html=True,
    )


# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>🖼️ Product Image Category Classifier</h1>
    <p>Upload product images and get instant category predictions powered by MobileNetV2 Transfer Learning</p>
</div>
""", unsafe_allow_html=True)


# ─── Model Loading ───
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH))


def preprocess_image(img: Image.Image):
    img_resized = img.resize(IMG_SIZE)
    img_array   = np.array(img_resized).astype("float32") / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    return np.expand_dims(img_array, axis=0)


def predict_category(model, img: Image.Image):
    processed  = preprocess_image(img)
    preds      = model.predict(processed, verbose=0).flatten()
    sorted_idx = np.argsort(preds)[::-1]
    return [
        {
            "Category": CLASS_NAMES[idx],
            "Probability": float(preds[idx]),
            "Icon": CATEGORY_ICONS.get(CLASS_NAMES[idx], "📦"),
        }
        for idx in sorted_idx
    ]


# ─── Tabs ───
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Single Prediction", "📁 Batch Prediction",
    "📊 Model Info",        "📋 Project Summary",
])

# ═══════════════════════════════════════════
# TAB 1 — Single Prediction
# ═══════════════════════════════════════════
with tab1:
    st.markdown("#### Upload a product image to get an instant category prediction")

    model = load_model()
    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png", "webp"],
        key="single"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col_result:
            if model is not None:
                with st.spinner("🔮 Predicting..."):
                    results = predict_category(model, img)

                top   = results[0]
                icon  = top["Icon"]
                name  = top["Category"].replace("_", " ").title()
                prob  = top["Probability"]
                color = CATEGORY_COLORS.get(top["Category"], "#8b5cf6")

                # Main result card
                confidence_ok = prob >= confidence_threshold / 100
                st.markdown(f"""
                <div class="result-card">
                    <div class="icon">{icon}</div>
                    <div class="name">{name}</div>
                    <div class="score">{prob:.1%}</div>
                    <div class="label">{"✅ High confidence" if confidence_ok else "⚠️ Low confidence"}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Top {top_k} Predictions:**")

                top_results = results[:top_k]
                fig = go.Figure(go.Bar(
                    x=[r["Probability"] for r in top_results],
                    y=[f"{r['Icon']} {r['Category'].replace('_', ' ')}" for r in top_results],
                    orientation='h',
                    marker_color=[CATEGORY_COLORS.get(r["Category"], "#8b5cf6") for r in top_results],
                    text=[f"{r['Probability']:.1%}" for r in top_results],
                    textposition='inside',
                ))
                fig.update_layout(
                    template="plotly_dark", height=280,
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Model not loaded. Copy `mobilenetv2_local.h5` to the app folder.")

    else:
        # Category showcase
        st.markdown("### 📦 Supported Categories")
        cols = st.columns(3)
        for i, name in enumerate(CLASS_NAMES):
            with cols[i % 3]:
                icon    = CATEGORY_ICONS.get(name, "📦")
                display = name.replace("_", " ").title()
                color   = CATEGORY_COLORS.get(name, "#8b5cf6")
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{color}30,{color}10);
                    padding:1rem; border-radius:12px; text-align:center;
                    margin-bottom:0.5rem; border-left:4px solid {color};">
                    <h3 style="margin:0">{icon}</h3>
                    <p style="margin:0;font-weight:600;font-size:0.85rem;color:#e2e8f0">{display}</p>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2 — Batch Prediction
# ═══════════════════════════════════════════
with tab2:
    st.markdown("#### Upload multiple product images for bulk category prediction")

    uploaded_files = st.file_uploader(
        "Upload product images (multiple)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch"
    )

    if uploaded_files:
        model = load_model()
        if model is None:
            st.error("❌ Model not loaded.")
        else:
            if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                all_results = []
                progress    = st.progress(0)
                status      = st.empty()

                for idx, file in enumerate(uploaded_files):
                    status.text(f"Processing {file.name} ({idx+1}/{len(uploaded_files)})...")
                    img     = Image.open(file).convert("RGB")
                    results = predict_category(model, img)
                    top     = results[0]
                    all_results.append({
                        "File":           file.name,
                        "Prediction":     top["Category"].replace("_", " ").title(),
                        "Category Code":  top["Category"],
                        "Confidence":     top["Probability"],
                        "Icon":           top["Icon"],
                    })
                    progress.progress((idx + 1) / len(uploaded_files))

                status.empty()
                results_df = pd.DataFrame(all_results)

                # KPIs
                c1, c2, c3 = st.columns(3)
                high_conf = (results_df["Confidence"] >= confidence_threshold / 100).sum()
                with c1:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-v">{len(results_df)}</div><div class="kpi-l">Total Images</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-v">{results_df["Confidence"].mean():.1%}</div><div class="kpi-l">Avg Confidence</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-v">{high_conf}</div><div class="kpi-l">High Confidence (≥{confidence_threshold}%)</div></div>', unsafe_allow_html=True)

                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    cat_counts = results_df["Prediction"].value_counts()
                    fig_dist   = go.Figure(data=[go.Pie(
                        labels=cat_counts.index, values=cat_counts.values,
                        marker=dict(colors=[CATEGORY_COLORS.get(k.upper().replace(" ", "_"), "#8b5cf6") for k in cat_counts.index]),
                        hole=0.4,
                    )])
                    fig_dist.update_layout(title="Category Distribution", height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col2:
                    fig_conf = go.Figure()
                    fig_conf.add_trace(go.Histogram(x=results_df["Confidence"], nbinsx=20, marker_color="#8b5cf6"))
                    fig_conf.add_vline(x=confidence_threshold / 100, line_dash="dash", line_color="#f87171",
                                       annotation_text=f"Threshold: {confidence_threshold}%")
                    fig_conf.update_layout(title="Confidence Score Distribution", template="plotly_dark",
                                           height=380, xaxis=dict(tickformat=".0%"),
                                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_conf, use_container_width=True)

                # Image gallery
                st.markdown("### 🖼️ Results Gallery")
                gallery_cols = st.columns(4)
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    with gallery_cols[idx % 4]:
                        img = Image.open(uploaded_files[idx]).convert("RGB")
                        st.image(img, use_container_width=True)
                        ok = row["Confidence"] >= confidence_threshold / 100
                        c  = "#4ade80" if ok else "#fb923c"
                        st.markdown(f'<p style="margin:2px 0;font-weight:700;font-size:0.85rem">{row["Icon"]} {row["Prediction"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="margin:0;color:{c};font-size:0.82rem;font-weight:600">{row["Confidence"]:.1%}</p>', unsafe_allow_html=True)

                with st.expander("📋 Full Results Table"):
                    display_df = results_df[["File", "Prediction", "Confidence"]].copy()
                    display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Upload multiple product images to run bulk prediction.")

# ═══════════════════════════════════════════
# TAB 3 — Model Info
# ═══════════════════════════════════════════
with tab3:
    st.markdown("#### MobileNetV2 Transfer Learning — Architecture & Training Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🤖 Model Architecture

        | Property | Value |
        |----------|-------|
        | **Base Model** | MobileNetV2 (ImageNet) |
        | **Input Size** | 224 × 224 × 3 |
        | **Output Classes** | 9 |
        | **Top Layers** | Flatten → Dense(512) → BN → Dropout(0.5) → Dense(256) → BN → Dropout(0.5) → Softmax |
        | **Optimizer** | Adam (lr = 1e-4) |
        | **Loss Function** | Categorical Crossentropy |
        | **Augmentation** | Horizontal Flip, Zoom(0.2), Shear(0.2) |
        | **Callbacks** | EarlyStopping (patience=5), ReduceLROnPlateau |
        | **Max Epochs** | 20 |
        """)

    with col2:
        st.markdown("### 📦 Supported Categories")
        for name in CLASS_NAMES:
            icon    = CATEGORY_ICONS.get(name, "📦")
            display = name.replace("_", " ").title()
            color   = CATEGORY_COLORS.get(name, "#8b5cf6")
            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:0.45rem 0.8rem;
                margin-bottom:5px;border-radius:8px;
                background:{color}18;border-left:3px solid {color};">
                <span style="font-size:1.3rem;margin-right:10px">{icon}</span>
                <span style="font-weight:600;color:#e2e8f0">{display}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ Model Pipeline")

    steps    = ["Input\nImage", "Resize\n224×224", "Normalize\n÷255", "MobileNetV2\n(Frozen)",
                "Flatten", "Dense\n512+BN", "Dense\n256+BN", "Softmax\n9 Classes"]
    colors_p = ["#3b82f6","#6366f1","#8b5cf6","#a855f7","#c084fc","#d8b4fe","#818cf8","#10b981"]

    fig_pipeline = go.Figure()
    for i, (step, color) in enumerate(zip(steps, colors_p)):
        fig_pipeline.add_trace(go.Scatter(
            x=[i], y=[0], mode='markers+text',
            marker=dict(size=64, color=color, line=dict(width=2, color='white')),
            text=step, textposition='middle center',
            textfont=dict(size=9, color='white'),
            showlegend=False,
        ))
        if i < len(steps) - 1:
            fig_pipeline.add_annotation(
                x=i + 0.5, y=0, text="→", showarrow=False,
                font=dict(size=22, color='#8b5cf6'),
            )

    fig_pipeline.update_layout(
        template="plotly_dark", height=160,
        xaxis=dict(visible=False, range=[-0.5, len(steps) - 0.5]),
        yaxis=dict(visible=False, range=[-0.6, 0.6]),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pipeline, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 4 — Project Summary
# ═══════════════════════════════════════════
with tab4:
    st.markdown("#### Project Overview & Business Value")

    st.markdown("""
    ### 🎯 Product Image Category Classification (Computer Vision)

    **Objective:** Automatically classify Amazon product images into the correct category to
    speed up listing workflows and reduce manual categorization errors.

    ---

    #### 📌 Approach: Transfer Learning with MobileNetV2

    **Why MobileNetV2?**
    - Lightweight and fast (mobile-friendly architecture)
    - Pre-trained on ImageNet (1.4M images, 1,000 classes) — strong visual feature extractor
    - Depthwise separable convolutions for high efficiency with low parameter count
    - Well-suited for fine-tuning on domain-specific datasets

    ---

    #### 📊 Pipeline
    1. **Data Preparation** → Train / Val / Check folders (9 categories)
    2. **Augmentation** → Horizontal flip, zoom (0.2), shear (0.2) to improve generalization
    3. **Transfer Learning** → MobileNetV2 base (frozen weights) + custom top layers
    4. **Training** → Up to 20 epochs, EarlyStopping, ReduceLROnPlateau
    5. **Evaluation** → Confusion matrix, classification report, per-class accuracy
    6. **Deployment** → Real-time prediction via Streamlit (single & batch modes)

    ---

    #### 📦 9 Product Categories
    """)

    cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        with cols[i % 3]:
            icon    = CATEGORY_ICONS.get(name, "📦")
            color   = CATEGORY_COLORS.get(name, "#8b5cf6")
            display = name.replace("_", " ").title()
            st.markdown(f"""
            <div style="background:{color}18;border-left:4px solid {color};
                padding:0.6rem 0.9rem;border-radius:8px;margin-bottom:6px;">
                {icon} <strong style="color:#e2e8f0">{display}</strong>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    ---

    #### 💡 Business Value
    | Benefit | Impact |
    |---------|--------|
    | **Automatic Categorization** | Instantly classify new products without manual review |
    | **Faster Listing** | Reduce time sellers spend on product categorization |
    | **Error Reduction** | Detect and flag misclassified products |
    | **Scalability** | Classify thousands of products in seconds with batch mode |
    | **Consistency** | Eliminate human variability in category assignment |

    ---
    """)

    st.caption("🔧 Deployment: Streamlit  |  Model: MobileNetV2 Transfer Learning  |  9 Product Categories")
