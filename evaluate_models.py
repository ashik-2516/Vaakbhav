"""
VaakBhav — evaluate_models.py
Research-Grade Model Evaluation Bench with Stratified 5-Fold Cross Validation
Provides leak-free comparative performance across 5 model architectures.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from train_model import clean_text, CSV_PATH

def run_evaluation():
    print("Loading dataset for 5-fold Stratified Cross-Validation...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    text_col = next((c for c in df.columns if 'text' in c), None)
    label_col = next((c for c in df.columns if 'label' in c or 'sentiment' in c), None)
    
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    df = df.dropna().reset_index(drop=True)

    le = LabelEncoder()
    try:
        y = le.fit_transform(df['label'].astype(int))
    except Exception:
        y = le.fit_transform(df['label'])

    print(f"Total samples: {len(df)}")
    print("Preprocessing text...")
    X_clean = [clean_text(t) for t in df['text']]

    word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=15000, sublinear_tf=True)

    X_word = word_vec.fit_transform(X_clean)
    X_char = char_vec.fit_transform(X_clean)
    X_combined = hstack([X_word, X_char]).tocsr()

    print(f"Feature matrix shape: {X_combined.shape}")

    models = {
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=42), method="sigmoid"),
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", C=2.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Soft Voting Ensemble": VotingClassifier([
            ("svm", CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=42), method="sigmoid")),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", C=2.0, random_state=42)),
            ("nb", MultinomialNB(alpha=0.1))
        ], voting="soft")
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro"
    }

    results = {}
    print("\nExecuting 5-Fold Stratified Cross-Validation...")
    print(f"{'Model Architecture':<30s} | {'Accuracy':<10s} | {'Macro F1':<10s} | {'Weighted F1':<12s} | {'Precision':<10s} | {'Recall':<10s}")
    print("-" * 95)

    for name, clf in models.items():
        cv_res = cross_validate(clf, X_combined, y, cv=skf, scoring=scoring, n_jobs=-1)
        acc_mean = float(np.mean(cv_res['test_accuracy']) * 100)
        macro_f1_mean = float(np.mean(cv_res['test_macro_f1']) * 100)
        weighted_f1_mean = float(np.mean(cv_res['test_weighted_f1']) * 100)
        prec_mean = float(np.mean(cv_res['test_precision_macro']) * 100)
        rec_mean = float(np.mean(cv_res['test_recall_macro']) * 100)

        results[name] = {
            "cv_accuracy_mean": round(acc_mean, 2),
            "cv_accuracy_std": round(float(np.std(cv_res['test_accuracy']) * 100), 2),
            "cv_macro_f1_mean": round(macro_f1_mean, 2),
            "cv_macro_f1_std": round(float(np.std(cv_res['test_macro_f1']) * 100), 2),
            "cv_weighted_f1_mean": round(weighted_f1_mean, 2),
            "cv_precision_mean": round(prec_mean, 2),
            "cv_recall_mean": round(rec_mean, 2)
        }

        print(f"{name:<30s} | {acc_mean:6.2f}%    | {macro_f1_mean:6.2f}%    | {weighted_f1_mean:8.2f}%    | {prec_mean:7.2f}%   | {rec_mean:6.2f}%")

    with open("models/cv_evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n5-Fold CV report saved to models/cv_evaluation_report.json")

if __name__ == "__main__":
    run_evaluation()
