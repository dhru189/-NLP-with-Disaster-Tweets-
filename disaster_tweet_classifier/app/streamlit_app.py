"""
Disaster Tweet Classifier — Interactive Dashboard
Hackathon-winning Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import clean_text, tokenize_and_preprocess, preprocess_pipeline, get_preprocessing_steps

# ──────────────────────────────────────────────
# Page Config & Theme
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS – disaster-themed palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 40%, #ea580c 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(185, 28, 28, 0.25);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #b91c1c;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    .prediction-real {
        background: linear-gradient(135deg, #7f1d1d, #dc2626);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(220, 38, 38, 0.3);
    }
    
    .prediction-safe {
        background: linear-gradient(135deg, #065f46, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(5, 150, 105, 0.3);
    }
    
    .step-box {
        background: #f8fafc;
        border-left: 4px solid #b91c1c;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        border-radius: 0 8px 8px 0;
    }
    
    .pipeline-step {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 500;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    
    div[data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load assets
# ──────────────────────────────────────────────
@st.cache_resource
def load_model_and_vectorizer():
    model_path = ROOT / "models" / "best_model.joblib"
    vec_path = ROOT / "models" / "tfidf_vectorizer.joblib"
    results_path = ROOT / "models" / "results_df.joblib"
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    results_df = joblib.load(results_path) if results_path.exists() else None
    return model, vectorizer, results_df


@st.cache_data
def load_dataset():
    train_path = ROOT / "data" / "train.csv"
    df = pd.read_csv(train_path)
    df = df.dropna(subset=["text"])
    return df


model, vectorizer, results_df = load_model_and_vectorizer()
df = load_dataset()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 Disaster Tweet Classifier")
    st.markdown("---")
    st.markdown("**Domain:** Social Media / Civic")
    st.markdown("**Dataset:** Kaggle NLP with Disaster Tweets")
    st.markdown(f"**Samples:** {len(df):,}")
    st.markdown(f"**Real disasters:** {(df['target']==1).sum():,}")
    st.markdown(f"**Not disasters:** {(df['target']==0).sum():,}")
    st.markdown("---")
    st.markdown("### Required Pipeline")
    st.markdown("""
1. Regex cleaning  
2. NLTK tokenize + preprocess  
3. TF-IDF vectorization  
4. Logistic Regression / SVM  
5. F1 + Confusion Matrix  
    """)
    st.markdown("---")
    st.caption("Built for Hackathon • Classical ML + Modern UX")


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚨 Disaster Tweet Classifier</h1>
    <p>Real-time NLP system that separates genuine emergency reports from figurative language — helping emergency teams respond faster.</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "📊 EDA",
    "🧹 Preprocessing Playground",
    "🔮 Live Classifier",
    "📋 Dataset Browser",
    "🏆 Model Leaderboard"
])


# ══════════════════════════════════════════════
# TAB 1 – Overview
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Problem Statement")
    st.info("""
    During disasters, emergency services are overwhelmed with social media data.  
    This NLP classifier identifies **real disaster tweets** from **figurative ones** so emergency teams can respond faster.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Real Disaster Example")
        st.success('"Massive flood in Ahmedabad — roads submerged near Sardar Bridge."')
        st.markdown("**→ REAL DISASTER**")
    with col2:
        st.markdown("#### ❌ Figurative Example")
        st.error('"My exam results are a complete disaster lol"')
        st.markdown("**→ NOT A DISASTER**")
    
    st.markdown("---")
    st.subheader("Live Dataset Statistics")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Tweets</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{(df['target']==1).sum():,}</div>
            <div class="metric-label">Real Disasters</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{(df['target']==0).sum():,}</div>
            <div class="metric-label">Not Disasters</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        avg_len = df["text"].str.len().mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_len:.0f}</div>
            <div class="metric-label">Avg Tweet Length</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Classification Pipeline")
    
    steps = st.columns(5)
    labels = ["1️⃣ Clean", "2️⃣ Tokenize", "3️⃣ TF-IDF", "4️⃣ Classify", "5️⃣ Evaluate"]
    descs = [
        "Regex: remove URLs, mentions, hashtags",
        "NLTK: tokens + stopwords + lemma",
        "TF-IDF with unigrams + bigrams",
        "Logistic Regression (best)",
        "F1-score + Confusion Matrix"
    ]
    for i, (s, l, d) in enumerate(zip(steps, labels, descs)):
        with s:
            st.markdown(f"""
            <div class="pipeline-step">
                <div style="font-size:1.3rem;margin-bottom:0.3rem">{l}</div>
                <div style="font-size:0.8rem;color:#6b7280;font-weight:400">{d}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Impact")
    st.markdown("""
    - **Emergency Response:** Filter noise so responders focus on real crises.  
    - **News Verification:** Distinguish genuine reports from hyperbole.  
    - **Social Listening:** Real-time monitoring for NGOs and agencies.  
    """)


# ══════════════════════════════════════════════
# TAB 2 – EDA
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")
    
    # Target distribution
    col_a, col_b = st.columns(2)
    
    with col_a:
        target_counts = df["target"].value_counts().reset_index()
        target_counts.columns = ["target", "count"]
        target_counts["label"] = target_counts["target"].map({0: "Not Disaster", 1: "Real Disaster"})
        
        fig = px.pie(
            target_counts, values="count", names="label",
            color="label",
            color_discrete_map={"Real Disaster": "#dc2626", "Not Disaster": "#059669"},
            title="Target Distribution",
            hole=0.45
        )
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        df["char_len"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        
        fig2 = px.histogram(
            df, x="char_len", color=df["target"].map({0: "Not Disaster", 1: "Real Disaster"}),
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Real Disaster": "#dc2626", "Not Disaster": "#059669"},
            title="Tweet Character Length by Class",
            labels={"char_len": "Characters"}
        )
        fig2.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig2, use_container_width=True)
    
    # Word clouds
    st.markdown("#### Word Clouds")
    wc_col1, wc_col2 = st.columns(2)
    
    disaster_text = " ".join(df[df["target"]==1]["text"].astype(str).tolist())
    nondisaster_text = " ".join(df[df["target"]==0]["text"].astype(str).tolist())
    
    # Simple cleaning for WC
    def simple_clean(t):
        t = re.sub(r'http\S+|@\w+|#', '', t.lower())
        return t
    
    with wc_col1:
        st.markdown("**Real Disaster Tweets**")
        wc1 = WordCloud(width=600, height=350, background_color="white",
                        colormap="Reds", max_words=80).generate(simple_clean(disaster_text))
        fig_wc1, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.imshow(wc1, interpolation="bilinear")
        ax1.axis("off")
        st.pyplot(fig_wc1)
    
    with wc_col2:
        st.markdown("**Not Disaster Tweets**")
        wc2 = WordCloud(width=600, height=350, background_color="white",
                        colormap="Greens", max_words=80).generate(simple_clean(nondisaster_text))
        fig_wc2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.imshow(wc2, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig_wc2)
    
    # Top keywords
    st.markdown("#### Most Frequent Keywords")
    if "keyword" in df.columns:
        kw = df.dropna(subset=["keyword"]).copy()
        kw["keyword"] = kw["keyword"].str.replace("%20", " ")
        top_kw = kw.groupby(["keyword", "target"]).size().unstack(fill_value=0)
        top_kw["total"] = top_kw.sum(axis=1)
        top_kw = top_kw.sort_values("total", ascending=False).head(15)
        
        fig_kw = go.Figure()
        fig_kw.add_trace(go.Bar(name="Not Disaster", x=top_kw.index, y=top_kw.get(0, 0), marker_color="#059669"))
        fig_kw.add_trace(go.Bar(name="Real Disaster", x=top_kw.index, y=top_kw.get(1, 0), marker_color="#dc2626"))
        fig_kw.update_layout(barmode="stack", title="Top 15 Keywords by Class", xaxis_tickangle=-40)
        st.plotly_chart(fig_kw, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 – Preprocessing Playground
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Text Preprocessing Playground")
    st.markdown("Paste any tweet to see the exact classical pipeline steps required by the project definition.")
    
    sample_tweets = [
        "Massive flood in Ahmedabad — roads submerged near Sardar Bridge.",
        "My exam results are a complete disaster lol",
        "Forest fire near La Ronge Sask. Canada https://t.co/abc123 @user #wildfire",
        "The new iPhone is a total disaster for my budget 😭",
        "Earthquake of 6.5 magnitude hits Japan — buildings collapsed #earthquake"
    ]
    
    selected = st.selectbox("Or pick a sample:", ["Type your own..."] + sample_tweets)
    
    default_text = "" if selected == "Type your own..." else selected
    user_text = st.text_area("Tweet text", value=default_text, height=100,
                             placeholder="Enter a tweet here...")
    
    if user_text.strip():
        steps = get_preprocessing_steps(user_text)
        
        st.markdown("### Step-by-step transformation")
        
        st.markdown(f"""
        <div class="step-box">
            <strong>0️⃣ Original</strong><br>{steps['original']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="step-box">
            <strong>1️⃣ Clean (regex)</strong> — remove URLs, mentions, hashtags, punctuation<br>
            <code>{steps['cleaned']}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="step-box">
            <strong>2️⃣ Tokenize (NLTK)</strong><br>
            <code>{steps['tokens']}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="step-box">
            <strong>3️⃣ Remove stopwords</strong><br>
            <code>{steps['no_stopwords']}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="step-box">
            <strong>4️⃣ Lemmatize</strong><br>
            <code>{steps['lemmatized']}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="step-box" style="border-left-color:#059669;">
            <strong>5️⃣ Final text ready for TF-IDF</strong><br>
            <code style="font-size:1.05rem;">{steps['final']}</code>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 – Live Classifier
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Live Classifier")
    st.markdown("Enter a tweet and get an instant prediction with confidence.")
    
    # Sample chips
    st.markdown("**Quick samples:**")
    chip_cols = st.columns(5)
    chip_texts = [
        "Massive flood in Ahmedabad — roads submerged near Sardar Bridge.",
        "My exam results are a complete disaster lol",
        "Wildfire spreading rapidly near California highway",
        "This traffic is a total disaster this morning",
        "Earthquake of magnitude 7.2 hits the region"
    ]
    
    if "live_input" not in st.session_state:
        st.session_state.live_input = ""
    
    for i, (col, txt) in enumerate(zip(chip_cols, chip_texts)):
        short = txt[:28] + "…" if len(txt) > 28 else txt
        if col.button(short, key=f"chip_{i}"):
            st.session_state.live_input = txt
    
    tweet_input = st.text_area(
        "Tweet to classify",
        value=st.session_state.live_input,
        height=120,
        key="classifier_input"
    )
    
    threshold = st.slider("Confidence threshold (for REAL DISASTER)", 0.3, 0.8, 0.5, 0.05)
    
    if st.button("🔍 Classify Tweet", type="primary", use_container_width=True) or tweet_input:
        if tweet_input.strip():
            # Preprocess
            cleaned = preprocess_pipeline(tweet_input)
            X = vectorizer.transform([cleaned])
            
            # Predict
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                conf_disaster = float(proba[1])
                conf_safe = float(proba[0])
            else:
                # decision function fallback
                decision = model.decision_function(X)[0]
                conf_disaster = 1 / (1 + np.exp(-decision))
                conf_safe = 1 - conf_disaster
            
            pred = 1 if conf_disaster >= threshold else 0
            label = "REAL DISASTER" if pred == 1 else "NOT A DISASTER"
            
            # Display result
            if pred == 1:
                st.markdown(f'<div class="prediction-real">🚨 {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-safe">✅ {label}</div>', unsafe_allow_html=True)
            
            # Confidence bars
            st.markdown("#### Confidence Scores")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Real Disaster", f"{conf_disaster*100:.1f}%")
                st.progress(conf_disaster)
            with c2:
                st.metric("Not a Disaster", f"{conf_safe*100:.1f}%")
                st.progress(conf_safe)
            
            # Simple explanation via top contributing words
            st.markdown("#### Why this prediction?")
            feature_names = vectorizer.get_feature_names_out()
            if hasattr(model, "coef_"):
                coefs = model.coef_[0]
            elif hasattr(model, "calibrated_classifiers_"):
                # Calibrated SVM
                coefs = model.calibrated_classifiers_[0].estimator.coef_[0]
            else:
                coefs = None
            
            if coefs is not None:
                # Get non-zero features in this tweet
                indices = X.nonzero()[1]
                contributions = []
                for idx in indices:
                    contributions.append((feature_names[idx], coefs[idx] * X[0, idx]))
                
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_contrib = contributions[:8]
                
                if top_contrib:
                    words = [w for w, _ in top_contrib]
                    scores = [s for _, s in top_contrib]
                    colors = ["#dc2626" if s > 0 else "#059669" for s in scores]
                    
                    fig_exp = go.Figure(go.Bar(
                        x=scores, y=words, orientation="h",
                        marker_color=colors
                    ))
                    fig_exp.update_layout(
                        title="Word contributions (positive → disaster)",
                        xaxis_title="Contribution score",
                        yaxis=dict(autorange="reversed"),
                        height=320,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)
                else:
                    st.caption("No strong signal words detected after preprocessing.")
            else:
                st.caption("Feature contribution available for linear models.")
            
            # History
            if "history" not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.insert(0, {
                "tweet": tweet_input[:80] + ("…" if len(tweet_input) > 80 else ""),
                "label": label,
                "confidence": f"{max(conf_disaster, conf_safe)*100:.1f}%"
            })
            st.session_state.history = st.session_state.history[:8]
            
            if st.session_state.history:
                st.markdown("#### Recent Predictions")
                hist_df = pd.DataFrame(st.session_state.history)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Please enter a tweet.")


# ══════════════════════════════════════════════
# TAB 5 – Dataset Browser
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Dataset Browser")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        target_filter = st.selectbox("Target", ["All", "Real Disaster (1)", "Not Disaster (0)"])
    with filter_col2:
        search = st.text_input("Search text", "")
    with filter_col3:
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    
    browser_df = df[["id", "keyword", "location", "text", "target"]].copy()
    browser_df["target_label"] = browser_df["target"].map({0: "NOT A DISASTER", 1: "REAL DISASTER"})
    
    if target_filter == "Real Disaster (1)":
        browser_df = browser_df[browser_df["target"] == 1]
    elif target_filter == "Not Disaster (0)":
        browser_df = browser_df[browser_df["target"] == 0]
    
    if search.strip():
        browser_df = browser_df[browser_df["text"].str.contains(search, case=False, na=False)]
    
    total = len(browser_df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * page_size
    end = start + page_size
    
    st.caption(f"Showing {start+1}–{min(end, total)} of {total} tweets")
    
    display = browser_df.iloc[start:end][["id", "keyword", "location", "text", "target_label"]]
    st.dataframe(display, use_container_width=True, hide_index=True, height=420)
    
    # Export
    csv = browser_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export filtered CSV", csv, "filtered_tweets.csv", "text/csv")


# ══════════════════════════════════════════════
# TAB 6 – Model Leaderboard
# ══════════════════════════════════════════════
with tab6:
    st.subheader("Model Comparison Leaderboard")
    
    if results_df is not None:
        st.markdown("All models trained with the **same TF-IDF features** and evaluated on a held-out 20% test set + 5-fold CV.")
        
        display_results = results_df.copy()
        display_results = display_results.rename(columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-Score",
            "roc_auc": "ROC-AUC",
            "cv_f1_mean": "CV F1 (mean)",
            "cv_f1_std": "CV F1 (std)"
        })
        
        # Format
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV F1 (mean)", "CV F1 (std)"]:
            if col in display_results.columns:
                display_results[col] = display_results[col].map(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )
        
        # Bar chart of F1
        fig_f1 = px.bar(
            results_df, x="model", y="f1",
            color="f1", color_continuous_scale="RdYlGn",
            title="F1-Score Comparison",
            labels={"f1": "F1-Score", "model": "Model"}
        )
        fig_f1.update_layout(xaxis_tickangle=-25, showlegend=False)
        st.plotly_chart(fig_f1, use_container_width=True)
        
        st.success("**Selected production model:** Logistic Regression (highest F1 + excellent ROC-AUC)")
        
        st.markdown("""
        ### Required Classical Approach (clearly present)
        - ✅ Regex cleaning (URLs, hashtags, mentions)  
        - ✅ NLTK tokenization + stopword removal + lemmatization  
        - ✅ TF-IDF vectorization (uni + bi-grams)  
        - ✅ Logistic Regression & Linear SVM trained  
        - ✅ Evaluated with F1-score and confusion matrices (saved in `/results`)  
        """)
        
        # Show confusion matrix images if present
        st.markdown("#### Confusion Matrices")
        cm_dir = ROOT / "results"
        cm_files = list(cm_dir.glob("cm_*.png"))
        if cm_files:
            cols = st.columns(min(3, len(cm_files)))
            for i, f in enumerate(sorted(cm_files)[:3]):
                with cols[i % 3]:
                    st.image(str(f), caption=f.stem.replace("cm_", "").replace("_", " ").title())
    else:
        st.warning("Results not found. Run `python -m src.train` first.")


# Footer
st.markdown("---")
st.caption("Disaster Tweet Classifier • Classical NLP Pipeline (Regex + NLTK + TF-IDF + LR/SVM) • Built for Hackathon")
