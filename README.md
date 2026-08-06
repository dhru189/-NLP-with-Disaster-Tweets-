# -NLP-with-Disaster-Tweets-
**Real disaster reports vs. figurative language — separated by an NLP pipeline that runs live in your browser.**

> *"Massive flood in Ahmedabad — roads submerged near Sardar Bridge."* → **REAL DISASTER**
> *"My exam results are a complete disaster lol"* → **NOT A DISASTER**

Domain: Social Media / Civic · Dataset: [Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) (7,613 labeled tweets)

---

## The problem

During a disaster, emergency services are flooded with social media posts — most of which aren't emergencies at all. A keyword filter can't tell "fire" (a wildfire) from "fire" (a hot new album). This project builds an NLP classifier that reads past the keywords to the meaning, so response teams can triage signal instead of chatter.

## The solution

A complete, working ML pipeline — regex cleaning → NLTK preprocessing → TF-IDF → five benchmarked classifiers — packaged with a dashboard where the winning model runs **entirely client-side**, no backend, no API key, no network calls after page load.

## Demo

`dashboard/index.html` **is** the live demo — open it in any browser and every tab is fully functional immediately, including real ML inference in the "Live Classifier" tab. No deployment, server, or account needed.

(This build takes the original prototype's tab structure — Overview / EDA / Preprocessing / Classifier / Dataset Browser — and rebuilds it with a real trained model at every layer instead of placeholder data.)

## Screenshots

| Overview | Exploratory Analysis |
|---|---|
| ![Overview](assets/screenshots/01_overview.png) | ![EDA](assets/screenshots/02_eda.png) |

| Preprocessing Playground | Live Classifier |
|---|---|
| ![Playground](assets/screenshots/03_playground.png) | ![Classifier](assets/screenshots/04_classifier.png) |

| Dataset Browser |
|---|
| ![Browser](assets/screenshots/05_browser.png) |

## Architecture

![Architecture diagram](assets/architecture_diagram.png)

The Logistic Regression model's TF-IDF vocabulary, IDF weights, and coefficients are exported to a 140KB JSON file. The dashboard reimplements sklearn's exact TF-IDF math (smoothed IDF, L2 normalization) and sigmoid scoring in vanilla JavaScript — verified byte-for-byte against the Python model's `predict_proba` output before shipping.

## Results

Five models, identical TF-IDF features (unigrams + bigrams, 2,500 terms), evaluated on a held-out 20% split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** 🥇 | 81.4% | 82.1% | 72.3% | **76.9%** | 86.9% | 74.3% ± 1.8 |
| Multinomial Naive Bayes | 81.3% | 83.4% | 70.5% | 76.4% | 86.3% | 73.2% ± 1.7 |
| Linear SVM | 79.9% | 78.9% | 72.6% | 75.6% | 86.2% | 73.4% ± 1.4 |
| Random Forest | 79.4% | 77.9% | 72.8% | 75.3% | 85.0% | 72.6% ± 1.7 |
| Gradient Boosting | 77.3% | 84.8% | 57.3% | 68.4% | 83.1% | 66.4% ± 2.5 |

Logistic Regression wins on F1 and ROC-AUC while being the cheapest to run — the right tradeoff for a model that needs to execute client-side. Full confusion matrix, ROC curve, and feature-importance breakdown are in the notebook.

Why the linear models beat the tree ensembles here: TF-IDF vectors are sparse and high-dimensional, exactly the regime where linear decision boundaries tend to generalize better than trees splitting on individual features.

## Features implemented

**Core pipeline (required approach)**
- [x] Regex cleaning — strip URLs, `@mentions`, `#` symbols, HTML entities, punctuation
- [x] NLTK tokenization, stopword removal (negations preserved), WordNet lemmatization
- [x] TF-IDF vectorization (unigrams + bigrams, 2,500 features)
- [x] Logistic Regression / Linear SVM classifiers
- [x] F1-score + confusion matrix evaluation

**Stronger technical core**
- [x] +3 additional models: Multinomial Naive Bayes, Random Forest, Gradient Boosting
- [x] Full metrics suite: Accuracy, Precision, Recall, F1, ROC-AUC, classification report
- [x] 5-fold stratified cross-validation comparison table
- [x] Feature importance (top TF-IDF terms by Logistic Regression coefficient)

**EDA**
- [x] Target distribution, tweet length & word-count distributions by class
- [x] Top unigrams / bigrams / trigrams per class
- [x] Word clouds (real vs. not-real)
- [x] Keyword-field and location-field analysis

**Dashboard (5 tabs, all interactive)**
- [x] **Overview** — problem statement, input/output examples, live dataset stats, pipeline stepper, model leaderboard
- [x] **Exploratory Data Analysis** — interactive Chart.js charts, class filters, keyword rankings
- [x] **Text Preprocessing Playground** — step-by-step visualization of a tweet moving through the pipeline
- [x] **Live Classifier** — real TF-IDF+LR inference in-browser, confidence threshold slider, "why this prediction" word-level explanation, prediction history, batch mode, CSV export
- [x] **Dataset Browser** — searchable, filterable, paginated table over a 2,000-tweet sample; CSV export

**Extras**
- [x] Model comparison leaderboard
- [x] Explainability (signed per-word contribution bars)
- [x] Batch classification (multi-line input)
- [x] Confidence threshold slider (tune precision/recall live)
- [x] CSV export (classifier batch results *and* dataset browser results)
- [x] Fully offline-capable — Chart.js is vendored inline, no CDN dependency

**Not run in this pass, scoped for later:** a fine-tuned DistilBERT model. Classical ML was prioritized because it trains in under a second, needs no GPU, and is small enough (~140KB) to genuinely run client-side — see [Next steps](#next-steps).

## Project structure

```
disaster-tweet-classifier/
├── data/
│   └── train.csv                    # Kaggle "NLP with Disaster Tweets" (7,613 rows)
├── src/
│   ├── preprocessing.py             # regex cleaning + NLTK pipeline (shared logic)
│   ├── train.py                     # TF-IDF + 5 models + metrics + model export
│   └── eda.py                       # EDA stats, word clouds, chart generation
├── notebooks/
│   └── disaster_tweet_classifier.ipynb   # full annotated, pre-executed notebook
├── models/
│   └── production_model.json        # exported TF-IDF vocab/idf + LR coef/intercept
├── results/
│   ├── metrics.json                 # per-model accuracy/precision/recall/F1/AUC/confusion matrix
│   ├── cv_results.json              # 5-fold CV scores
│   ├── roc_curves.json              # ROC curve points per model
│   ├── feature_importance.json      # top positive/negative TF-IDF terms
│   ├── eda.json                     # all EDA aggregates powering the dashboard
│   └── dataset_sample.json          # 2,000-row stratified sample for the Dataset Browser
├── dashboard/
│   ├── index.html                   # ★ the built, self-contained dashboard (open this)
│   ├── template.html                # HTML/CSS shell with placeholder tokens
│   ├── app.js                       # all dashboard JS (charts, playground, classifier, browser)
│   ├── build_dashboard.py           # stitches template + parts + data → index.html
│   ├── parts/                       # per-tab HTML fragments
│   └── vendor/chart.umd.min.js      # vendored Chart.js (no CDN dependency)
├── assets/                          # generated charts, word clouds, screenshots, diagram
├── requirements.txt
├── PITCH.md                         # 60-second judge pitch
└── README.md
```

## How to run

### The dashboard (no setup required)
Open **`dashboard/index.html`** directly in any browser. It's a single self-contained file — no server, no build step, no internet connection needed after the page loads (Chart.js is bundled inline).

### The ML pipeline
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

cd src
python train.py     # trains all 5 models, writes results/ + models/production_model.json
python eda.py        # generates EDA stats + charts into results/ and assets/
```

### The notebook
```bash
jupyter notebook notebooks/disaster_tweet_classifier.ipynb
```
Ships pre-executed — every cell's output (charts, tables, metrics) is already visible without re-running anything.

### Rebuilding the dashboard after changing data/model/charts
```bash
cd dashboard
python build_dashboard.py    # re-reads results/*.json + models/*.json → index.html
```

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `NLTK` · `matplotlib` · `seaborn` · `wordcloud` · vanilla JS + `Chart.js` (vendored, no build tooling)

## Dataset

[Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) ("Real or Not?" competition dataset): 7,613 tweets with `id`, `keyword`, `location`, `text`, and `target` (1 = real disaster, 0 = not).

## Credits

- Dataset: Kaggle *NLP with Disaster Tweets* competition
- Presentation & structure inspired by [Hazrat-Ali9/Disaster-Tweet-Classification-Using-NLP-and-Machine-Learning](https://github.com/Hazrat-Ali9/Disaster-Tweet-Classification-Using-NLP-and-Machine-Learning)

## Next steps

- Fine-tune a lightweight transformer (DistilBERT) for the sarcasm/idiom cases TF-IDF still misses
- Geocode the `location` field for a map view
- Streaming ingestion from a live Twitter/X-style feed
- Calibrated confidence scores (Platt scaling) for the tree-based models
