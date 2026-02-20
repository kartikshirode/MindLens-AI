"""
MindLens-AI — Streamlit Demo App

Run:  streamlit run app/streamlit_app.py
"""

import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import numpy as np

from src.model import load_model
from src.preprocessing import clean_text
from src.explainability import explain_lime
from src.robustness import remove_keywords, synonym_replace
from src.trust import generate_trust_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MindLens-AI",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load artifacts (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model, vectorizer = load_model("data/processed/model_artifacts.joblib")
    try:
        bias_data = joblib.load("data/processed/bias_results.joblib")
        bias_fpr_gap = bias_data["bias_fpr_gap"]
    except FileNotFoundError:
        bias_fpr_gap = 0.0
    return model, vectorizer, bias_fpr_gap


model, vectorizer, bias_fpr_gap = load_artifacts()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🧠 MindLens-AI")
st.markdown("**Explainable Mental Health Risk Detection** — Enter text below to analyze.")

col1, col2 = st.columns([2, 1])

with col1:
    user_text = st.text_area(
        "Input text",
        height=180,
        placeholder="Type or paste a social media post here...",
    )

    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze and user_text.strip():
    cleaned = clean_text(user_text)

    if not cleaned.strip():
        st.warning("Text is empty after cleaning. Please enter meaningful text.")
    else:
        # --- Prediction ---
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        confidence = float(probs[pred])
        label = "Risk" if pred == 1 else "No Risk"

        with col2:
            st.subheader("Prediction")
            st.metric("Result", label)
            st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        # --- LIME Explanation ---
        st.subheader("📖 Explanation (LIME)")
        exp = explain_lime(model, vectorizer, cleaned, num_features=10)

        lime_list = exp.as_list()
        positive_words = [w for w, s in lime_list if s > 0]
        negative_words = [w for w, s in lime_list if s < 0]

        lime_col1, lime_col2 = st.columns(2)
        with lime_col1:
            st.markdown("**Words pushing toward Risk (red):**")
            for word, score in sorted(lime_list, key=lambda x: -x[1]):
                if score > 0:
                    st.markdown(f"- `{word}` → +{score:.3f}")
        with lime_col2:
            st.markdown("**Words pushing toward No Risk (green):**")
            for word, score in sorted(lime_list, key=lambda x: x[1]):
                if score < 0:
                    st.markdown(f"- `{word}` → {score:.3f}")

        # --- Perturbation Comparison ---
        st.subheader("🔄 Perturbation Comparison")
        show_pert = st.toggle("Show perturbation results", value=True)

        if show_pert:
            pert_col1, pert_col2 = st.columns(2)

            # Keyword removal
            kw_text = remove_keywords(cleaned)
            X_kw = vectorizer.transform([kw_text])
            kw_pred = model.predict(X_kw)[0]
            kw_prob = model.predict_proba(X_kw)[0]
            kw_label = "🔴 Risk" if kw_pred == 1 else "🟢 No Risk"

            with pert_col1:
                st.markdown("**Keyword Removal**")
                st.write(f"Prediction: {kw_label} (conf: {float(kw_prob[kw_pred]):.1%})")
                flipped = "Yes ⚠️" if kw_pred != pred else "No ✓"
                st.write(f"Prediction flipped: {flipped}")

            # Synonym replacement
            syn_text = synonym_replace(cleaned, n=3, rng_seed=42)
            X_syn = vectorizer.transform([syn_text])
            syn_pred = model.predict(X_syn)[0]
            syn_prob = model.predict_proba(X_syn)[0]
            syn_label = "🔴 Risk" if syn_pred == 1 else "🟢 No Risk"

            with pert_col2:
                st.markdown("**Synonym Replacement**")
                st.write(f"Prediction: {syn_label} (conf: {float(syn_prob[syn_pred]):.1%})")
                flipped = "Yes ⚠️" if syn_pred != pred else "No ✓"
                st.write(f"Prediction flipped: {flipped}")

        # --- Trust Score ---
        st.subheader("🛡️ Trust Score")
        report = generate_trust_report(cleaned, model, vectorizer, bias_fpr_gap)

        trust_color = {
            "High": "🟢", "Medium": "🟡", "Low": "🔴"
        }
        badge = trust_color.get(report["trust_label"], "⚪")

        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Trust Score", f"{report['trust_score']:.3f}")
        t_col2.metric("Trust Level", f"{badge} {report['trust_label']}")
        t_col3.metric("Flip Risk", f"{'Yes' if report['flip_rate'] > 0 else 'No'}")

elif analyze:
    st.warning("Please enter some text to analyze.")

# --- Footer ---
st.divider()
st.caption("MindLens-AI v1.0 — Explainable, Fair, and Robust Mental Health Risk Detection")
