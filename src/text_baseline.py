"""
Text-only baseline training using TF-IDF + Logistic Regression.
This often gives a strong baseline for sarcasm classification.
"""
import os
from typing import Dict, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from joblib import dump

from utils import setup_logging, save_results

logger = setup_logging(level="INFO")


def train_text_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict
) -> Dict:
    """Train and evaluate a TF-IDF + Logistic Regression baseline."""
    logger.info("=" * 70)
    logger.info("TEXT-ONLY BASELINE: TF-IDF + LOGISTIC REGRESSION")
    logger.info("=" * 70)

    # Prepare data
    X_train = train_df['SENTENCE'].fillna("").astype(str)
    y_train = train_df['Sarcasm'].astype(int)
    X_test = test_df['SENTENCE'].fillna("").astype(str)
    y_test = test_df['Sarcasm'].astype(int)

    # Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        max_df=0.9
    )

    logger.info("Fitting TF-IDF vectorizer...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Classifier
    clf = LogisticRegression(
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1
    )

    logger.info("Training Logistic Regression classifier...")
    clf.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = clf.predict(X_test_tfidf)
    y_proba = clf.predict_proba(X_test_tfidf)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }

    logger.info("=" * 70)
    logger.info("TEXT BASELINE RESULTS")
    logger.info("=" * 70)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("=" * 70)

    # Save artifacts
    models_dir = config['paths']['models_dir']
    results_dir = config['paths']['results_dir']
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    vectorizer_path = os.path.join(models_dir, "text_vectorizer.joblib")
    model_path = os.path.join(models_dir, "text_baseline.joblib")
    dump(vectorizer, vectorizer_path)
    dump(clf, model_path)

    # Save results
    results = {
        "metrics": metrics,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path
    }
    results_path = os.path.join(results_dir, "text_baseline_results.json")
    save_results(results, results_path)

    report_path = os.path.join(results_dir, "text_baseline_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TEXT-ONLY BASELINE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        cm = metrics["confusion_matrix"]
        f.write(f"  TN: {cm[0][0]}  FP: {cm[0][1]}\n")
        f.write(f"  FN: {cm[1][0]}  TP: {cm[1][1]}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))

    logger.info(f"Saved text baseline model to {model_path}")
    logger.info(f"Saved vectorizer to {vectorizer_path}")
    logger.info(f"Saved results to {results_path}")
    logger.info(f"Saved report to {report_path}")

    return results
