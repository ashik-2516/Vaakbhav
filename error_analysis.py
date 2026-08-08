"""
VaakBhav — error_analysis.py
Research-Grade Failure Mode & Error Analysis Pipeline
Extracts misclassified instances and categorizes them into linguistic error types.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

from train_model import clean_text, CSV_PATH, NEGATION_WORDS

def run_error_analysis():
    print("Running Error & Failure Mode Analysis...")
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

    target_names = [str(c).capitalize() for c in le.classes_]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['text'], y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_clean = [clean_text(t) for t in X_train_raw]
    X_test_clean  = [clean_text(t) for t in X_test_raw]

    word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=15000, sublinear_tf=True)

    X_tr_word = word_vec.fit_transform(X_train_clean)
    X_te_word = word_vec.transform(X_test_clean)
    X_tr_char = char_vec.fit_transform(X_train_clean)
    X_te_char = char_vec.transform(X_test_clean)

    X_tr = hstack([X_tr_word, X_tr_char]).tocsr()
    X_te = hstack([X_te_word, X_te_char]).tocsr()

    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=42), method="sigmoid")
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)

    errors = []
    test_indices = X_test_raw.index.tolist()

    for idx, (true_label, pred_label, proba, raw_text, clean_t) in enumerate(zip(y_test, y_pred, y_prob, X_test_raw, X_test_clean)):
        if true_label != pred_label:
            words = set(raw_text.lower().split())
            has_negation = bool(words & NEGATION_WORDS)
            is_short = len(words) <= 4
            
            error_category = "General Sentiment Misclassification"
            if has_negation:
                error_category = "Negation Boundary Failure"
            elif is_short:
                error_category = "Short Text Ambiguity"
            elif (true_label == 1 or pred_label == 1):
                error_category = "Neutral/Polarity Boundary Ambiguity"

            errors.append({
                "index": test_indices[idx],
                "raw_text": raw_text,
                "cleaned_text": clean_t,
                "true_label": target_names[true_label],
                "predicted_label": target_names[pred_label],
                "confidence": round(float(np.max(proba)), 3),
                "category": error_category
            })

    total_test = len(y_test)
    total_errors = len(errors)
    accuracy = round((1 - total_errors / total_test) * 100, 2)

    categories_count = {}
    for err in errors:
        cat = err["category"]
        categories_count[cat] = categories_count.get(cat, 0) + 1

    summary = {
        "total_test_samples": total_test,
        "total_errors": total_errors,
        "test_accuracy": accuracy,
        "error_category_breakdown": categories_count,
        "sample_error_cases": errors[:15]  # Top 15 representative cases
    }

    with open("models/error_analysis_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nError Analysis Complete!")
    print(f"   Total Test Samples: {total_test}")
    print(f"   Total Misclassified: {total_errors} ({100-accuracy:.2f}% error rate)")
    print("\n   Failure Mode Breakdown:")
    for cat, count in categories_count.items():
        pct = (count / total_errors) * 100
        print(f"     - {cat:<40s}: {count:3d} cases ({pct:.1f}%)")

    print("\nReport saved to models/error_analysis_report.json")

if __name__ == "__main__":
    run_error_analysis()
