"""
Disaster Tweet Classifier - Training & Evaluation Module
Required: TF-IDF + SVM / Logistic Regression
Bonus: MultinomialNB, RandomForest, GradientBoosting
Full metrics + 5-fold CV + feature importance
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess_pipeline

RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(train_path: str):
    df = pd.read_csv(train_path)
    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].apply(preprocess_pipeline)
    return df


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight="balanced", random_state=RANDOM_STATE, dual="auto"
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_test, y_test, model_name: str, is_svm: bool = False):
    y_pred = model.predict(X_test)
    
    # Probability / decision for ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        # Scale to 0-1 approx
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
    else:
        y_prob = y_pred.astype(float)
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0,
    }
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["NOT A DISASTER", "REAL DISASTER"])
    
    return metrics, cm, report, y_pred, y_prob


def plot_confusion_matrix(cm, model_name: str, save_path: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn_r",
                xticklabels=["NOT A DISASTER", "REAL DISASTER"],
                yticklabels=["NOT A DISASTER", "REAL DISASTER"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def get_top_features(vectorizer, model, n: int = 20):
    """Top positive / negative features for linear models."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        # For tree models we just take absolute importance
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-n:][::-1]
        return {
            "top_disaster": list(zip(feature_names[top_idx], importances[top_idx])),
            "top_not_disaster": []
        }
    else:
        return {"top_disaster": [], "top_not_disaster": []}
    
    top_pos_idx = np.argsort(coefs)[-n:][::-1]
    top_neg_idx = np.argsort(coefs)[:n]
    
    return {
        "top_disaster": list(zip(feature_names[top_pos_idx], coefs[top_pos_idx])),
        "top_not_disaster": list(zip(feature_names[top_neg_idx], coefs[top_neg_idx])),
    }


def train_and_evaluate(train_path: str, test_size: float = 0.2):
    print("Loading & preprocessing data...")
    df = load_data(train_path)
    
    X = df["clean_text"]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models = get_models()
    results = []
    best_f1 = -1
    best_model_name = None
    best_model = None
    best_vectorizer = vectorizer
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    print("\n" + "="*60)
    print("TRAINING & EVALUATING MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n→ Training {name}...")
        
        # 5-fold CV on training set
        cv_scores = cross_val_score(
            model, X_train_tfidf, y_train, cv=cv, scoring="f1", n_jobs=-1
        )
        
        model.fit(X_train_tfidf, y_train)
        
        is_svm = "SVM" in name
        metrics, cm, report, y_pred, y_prob = evaluate_model(
            model, X_test_tfidf, y_test, name, is_svm
        )
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()
        
        results.append(metrics)
        
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1-Score : {metrics['f1']:.4f}")
        print(f"  CV F1    : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        
        # Save confusion matrix
        cm_path = os.path.join(RESULTS_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
        plot_confusion_matrix(cm, name, cm_path)
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name
            best_model = model
    
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON (sorted by F1)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    print(f"\n★ Best model: {best_model_name} (F1 = {best_f1:.4f})")
    
    # Feature importance for best linear model
    if hasattr(best_model, "coef_"):
        top_feats = get_top_features(vectorizer, best_model, n=15)
        print("\nTop words indicating REAL DISASTER:")
        for w, c in top_feats["top_disaster"]:
            print(f"  {w:20s} {c:+.4f}")
        print("\nTop words indicating NOT A DISASTER:")
        for w, c in top_feats["top_not_disaster"]:
            print(f"  {w:20s} {c:+.4f}")
    
    # Save best model + vectorizer
    # For SVM we calibrate so we can get probabilities
    if "SVM" in best_model_name:
        calibrated = CalibratedClassifierCV(best_model, cv=3)
        calibrated.fit(X_train_tfidf, y_train)
        best_model = calibrated
    
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(results_df, os.path.join(MODELS_DIR, "results_df.joblib"))
    
    # Also save the Logistic Regression specifically (required baseline)
    lr = models["Logistic Regression"]
    lr.fit(X_train_tfidf, y_train)
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    
    print(f"\nModels saved to {MODELS_DIR}")
    
    return results_df, best_model_name, vectorizer, best_model


if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    train_and_evaluate(train_path)
