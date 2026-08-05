# 🚨 Disaster Tweet Classifier

**Domain:** Social Media / Civic  
**Hackathon-ready classical NLP + modern interactive dashboard**

---

## Problem

During disasters, emergency services are flooded with social media noise.  
Many tweets contain disaster-related words used **figuratively** (“my exam was a disaster”), while others report **real emergencies**.

This project builds an NLP classifier that reliably separates the two so responders can focus on genuine crises.

### Input / Output

| Example Tweet | Label |
|---------------|-------|
| *"Massive flood in Ahmedabad — roads submerged near Sardar Bridge."* | **REAL DISASTER** |
| *"My exam results are a complete disaster lol"* | **NOT A DISASTER** |

---

## Required Classical Pipeline (strictly followed)

1. **Clean tweets** with regex → remove URLs, mentions (`@`), hashtag symbols, HTML entities, punctuation  
2. **Tokenize & preprocess** with NLTK → tokenization, stop-word removal, lemmatization  
3. **TF-IDF vectorization** (unigrams + bigrams, 10 k features)  
4. **Train classifiers** → Logistic Regression (primary) + Linear SVM  
5. **Evaluate** with F1-score, confusion matrix, precision, recall, ROC-AUC + 5-fold CV  

### Additional models for comparison
- Multinomial Naive Bayes  
- Random Forest  
- Gradient Boosting  

---

## Results (held-out 20 % test set)

| Model                    | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV F1 (mean) |
|--------------------------|----------|-----------|--------|----------|---------|--------------|
| **Logistic Regression**  | 0.807    | 0.775     | 0.775  | **0.775**| 0.872   | 0.747        |
| Multinomial Naive Bayes  | 0.819    | 0.869     | 0.680  | 0.763    | 0.868   | 0.727        |
| Linear SVM               | 0.793    | 0.751     | 0.772  | 0.762    | 0.855   | 0.722        |
| Gradient Boosting        | 0.775    | 0.849     | 0.578  | 0.688    | 0.835   | 0.643        |
| Random Forest            | 0.773    | 0.844     | 0.580  | 0.687    | 0.847   | 0.646        |

**Production model:** Logistic Regression (best F1 + excellent calibration).

Top disaster-indicating words: `hiroshima`, `fire`, `wildfire`, `california`, `flood`, `earthquake`, `killed`, `bombing` …  
Top non-disaster words: `love`, `want`, `bag`, `ruin`, `wrecked` …

---

## Interactive Dashboard (Streamlit)

A modern single-page application with 6 tabs:

| Tab | Features |
|-----|----------|
| **Overview** | Problem statement, impact story, live stats cards, visual pipeline stepper |
| **EDA** | Target pie, length histograms, dual word-clouds, keyword analysis |
| **Preprocessing Playground** | Paste any tweet → see exact regex + NLTK steps |
| **Live Classifier** | Instant prediction + confidence + word-level explanation + history |
| **Dataset Browser** | Searchable, filterable, paginated table + CSV export |
| **Model Leaderboard** | Full comparison table + F1 bar chart + confusion matrices |

**Theme:** Deep reds/oranges for disaster, teal/green for safe. Fully responsive.

---

## Project Structure

```
disaster_tweet_classifier/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/
│   ├── preprocess.py      # Regex + NLTK pipeline
│   └── train.py           # TF-IDF + all models + evaluation
├── models/
│   ├── best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── logistic_regression.joblib
├── results/
│   ├── model_comparison.csv
│   └── cm_*.png           # Confusion matrices
├── app/
│   └── streamlit_app.py   # Full interactive dashboard
├── notebooks/             # (optional) EDA exploration
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (First time) Train models
cd disaster_tweet_classifier
PYTHONPATH=src python -m src.train

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

Open the URL shown by Streamlit (usually http://localhost:8501).

---

## Architecture

```
Tweet text
    │
    ▼
┌──────────────────────┐
│  Regex Cleaning      │  ← remove URLs / @ / # / HTML / punct
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  NLTK Preprocessing  │  ← tokenize + stopwords + lemmatize
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  TF-IDF Vectorizer   │  ← 10k features, 1-2 grams
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Logistic Regression │  ← production model
│  (or SVM / NB / RF)  │
└──────────┬───────────┘
           │
           ▼
   REAL DISASTER  /  NOT A DISASTER
   + confidence + word contributions
```

---

## 60-Second Pitch for Judges

> “During disasters, emergency teams drown in tweets. Many use words like ‘flood’ or ‘disaster’ just for emphasis.  
> We built a classical NLP pipeline that is transparent and production-ready: regex cleaning, NLTK preprocessing, TF-IDF, and Logistic Regression — exactly as required.  
> It reaches 0.78 F1 and 0.87 ROC-AUC.  
> On top of the solid baseline we added a beautiful Streamlit dashboard with live classification, word-level explanations, preprocessing playground, full EDA, and a model leaderboard.  
> The system is ready for emergency operations centers today — no black-box transformers needed, fully interpretable, and fast.”

---

## References

- Dataset: [Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)  
- Inspiration: classical ML notebooks + [Hazrat-Ali9/Disaster-Tweet-Classification](https://github.com/Hazrat-Ali9/Disaster-Tweet-Classification-Using-NLP-and-Machine-Learning)  
- Live demo pattern from previous prototype

---

*Built to win. Classical core + product-quality UX.*
