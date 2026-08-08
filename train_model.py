"""
VaakBhav — train_model.py
Research-Grade Training Pipeline for Hindi-English (Hinglish) Sentiment Analysis
Handles:
  1. Data leakage prevention (train/test split executed BEFORE augmentation & filtering)
  2. Sentiment-preserving NLP preprocessing (negation preservation, Devanagari script, character elongation)
  3. Feature Union (Subword/Char TF-IDF + Word TF-IDF)
  4. Multi-model training, calibration, and automated best-model selection
  5. Serialized artifacts generation in ./models/
"""

import os
import sys
import json
import re
import joblib
import pandas as pd
import numpy as np
import nltk

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# ── Safe NLTK Setup ──────────────────────────────────────────────────────────
for resource in ["stopwords", "vader_lexicon"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Fallback English Stopwords
_FALLBACK_STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','only','own','same','so','than','too','very','s','t',
    'can','will','just','should','now','d','ll','m','o','re','ve','y'
}

# Negation words MUST NOT be stripped during sentiment preprocessing
NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none',
    'nhi', 'nahi', 'na', 'mat', 'naa', 'nahin', 'ni',
    'don', 'dont', 'ain', 'aren', 'couldn', 'didn', 'doesn',
    'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn', 'needn',
    'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}

try:
    raw_stops = set(nltk.corpus.stopwords.words("english"))
    STOP_WORDS = raw_stops - NEGATION_WORDS
except Exception:
    STOP_WORDS = _FALLBACK_STOPWORDS - NEGATION_WORDS

# Hinglish term normalization dictionary
HINGLISH_MAP = {
    "accha": "good", "acha": "good", "achha": "good", "acchi": "good", "achhi": "good",
    "bakwas": "terrible", "bekar": "terrible", "bura": "terrible", "burii": "terrible",
    "mast": "excellent", "kamaal": "excellent", "kamal": "excellent",
    "pyaar": "love", "pyar": "love", "mohabbat": "love",
    "mehnga": "expensive", "sasta": "cheap",
    "faltu": "useless", "ghatiya": "terrible",
    "timepass": "boring", "superb": "excellent",
    "bahut": "very", "bohot": "very", "theek": "okay", "thik": "okay",
    "bilkul": "absolutely", "yaar": "friend", "pasand": "like",
    "behtareen": "best", "khaas": "special", "zabardast": "amazing",
    "shandar": "wonderful", "pagal": "crazy", "bewakoof": "stupid",
    "khushi": "happy", "dukh": "sad", "sundar": "beautiful",
}

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "output__1_.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    """
    Research-Grade Sentiment-Preserving NLP Preprocessor.
    - Lowers case & removes URLs
    - Normalizes character elongations (e.g. superrrrr -> superr, achaaaa -> achaa)
    - Retains Latin and Devanagari script characters (\u0900-\u097F)
    - Maps common Hinglish terms while preserving negations
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Normalize elongated character repeats (3+ occurrences to 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Retain Latin letters, Devanagari range, numbers, and basic spaces
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    words = []
    for w in text.split():
        if w in STOP_WORDS:
            continue
        mapped = HINGLISH_MAP.get(w, w)
        words.append(mapped)
    return " ".join(words) if words else text

def augment_text(text: str) -> str:
    """Light augmentation applied STRICTLY to training samples only."""
    augmentation_map = {
        "good": ["nice", "great", "awesome"],
        "bad":  ["worst", "poor", "terrible"],
        "love": ["like", "enjoy"],
        "excellent": ["fantastic", "amazing"],
    }
    words = text.split()
    out = []
    for w in words:
        if w in augmentation_map and np.random.rand() < 0.2:
            out.append(np.random.choice(augmentation_map[w]))
        else:
            out.append(w)
    return " ".join(out)

def main():
    print("Loading dataset...")
    if not os.path.exists(CSV_PATH):
        print(f"Dataset file '{CSV_PATH}' not found!")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    
    text_col = next((c for c in df.columns if 'text' in c), None)
    label_col = next((c for c in df.columns if 'label' in c or 'sentiment' in c), None)
    
    if text_col is None or label_col is None:
        print(f"Required columns missing. Found: {list(df.columns)}")
        sys.exit(1)

    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    df = df.dropna(subset=['text', 'label']).reset_index(drop=True)
    
    print(f"   Total valid records: {len(df)}")
    print(f"   Label distribution:\n{df['label'].value_counts()}")

    # Encode labels
    le = LabelEncoder()
    try:
        y_encoded = le.fit_transform(df['label'].astype(int))
    except (ValueError, TypeError):
        y_encoded = le.fit_transform(df['label'])

    df['label_encoded'] = y_encoded
    label_dict = dict(zip(le.transform(le.classes_), le.classes_))
    print(f"   Classes mapping: {label_dict}")

    # ── LEAK-FREE SPLIT ───────────────────────────────────────────────────────
    print("\nPerforming Leak-Free Stratified Train/Test Split (80/20)...")
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        df['text'], df['label_encoded'], test_size=0.2, stratify=df['label_encoded'], random_state=42
    )

    print("Preprocessing train and test sets...")
    X_train_clean = [clean_text(t) for t in X_train_raw]
    X_test_clean  = [clean_text(t) for t in X_test_raw]

    # Apply data augmentation ONLY to training set
    print("Augmenting training set ONLY (preventing data leakage)...")
    aug_train_texts = [augment_text(t) for t in X_train_clean]
    
    X_train_final = X_train_clean + aug_train_texts
    y_train_final = np.concatenate([y_train_raw.values, y_train_raw.values])

    print(f"   Training samples (original + augmented): {len(X_train_final)}")
    print(f"   Test samples (held-out, unaugmented):    {len(X_test_clean)}")

    # ── FEATURE EXTRACTION ────────────────────────────────────────────────────
    print("\nFitting Feature Matrix (Word TF-IDF + Character/Subword TF-IDF)...")
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 6), max_features=15000, sublinear_tf=True)

    X_train_word = word_vectorizer.fit_transform(X_train_final)
    X_test_word  = word_vectorizer.transform(X_test_clean)

    X_train_char = char_vectorizer.fit_transform(X_train_final)
    X_test_char  = char_vectorizer.transform(X_test_clean)

    X_train_vec = hstack([X_train_word, X_train_char]).tocsr()
    X_test_vec  = hstack([X_test_word,  X_test_char]).tocsr()

    print(f"   Combined Feature Shape (Train): {X_train_vec.shape}")
    print(f"   Combined Feature Shape (Test):  {X_test_vec.shape}")

    # ── MODEL TRAINING & EVALUATION BENCH ─────────────────────────────────────
    print("\nTraining and evaluating multi-model candidates...")

    lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=2.0, random_state=42)
    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000, random_state=42), method="sigmoid")
    nb = MultinomialNB(alpha=0.1)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    ensemble = VotingClassifier([("svm", svm), ("lr", lr), ("nb", nb)], voting="soft", n_jobs=-1)

    candidates = {
        "Linear SVM (Calibrated)": svm,
        "Logistic Regression": lr,
        "Multinomial Naive Bayes": nb,
        "Random Forest": rf,
        "Soft Voting Ensemble": ensemble
    }

    eval_results = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_model = None

    for name, clf in candidates.items():
        clf.fit(X_train_vec, y_train_final)
        y_pred = clf.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred) * 100
        macro_f1 = f1_score(y_test, y_pred, average="macro") * 100
        weighted_f1 = f1_score(y_test, y_pred, average="weighted") * 100
        
        eval_results[name] = {
            "accuracy": round(acc, 2),
            "macro_f1": round(macro_f1, 2),
            "weighted_f1": round(weighted_f1, 2)
        }
        
        print(f"   {name:30s} -> Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.2f}% | Weighted F1: {weighted_f1:.2f}%")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name
            best_model = clf

    print(f"\nSelected Best Model: {best_model_name} (Macro F1: {best_macro_f1:.2f}%)")

    # Full report on test set for best model
    best_y_pred = best_model.predict(X_test_vec)
    target_names = [str(label_dict.get(i, i)) for i in sorted(label_dict)]
    report_str = classification_report(y_test, best_y_pred, target_names=target_names)
    print("\nClassification Report (Held-out Test Set):")
    print(report_str)

    cm = confusion_matrix(y_test, best_y_pred).tolist()

    # ── SERIALIZATION ─────────────────────────────────────────────────────────
    print("\nSaving trained models and vectorizers to ./models/ ...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(word_vectorizer, os.path.join(MODEL_DIR, "word_vectorizer.pkl"))
    joblib.dump(char_vectorizer, os.path.join(MODEL_DIR, "char_vectorizer.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    metrics_payload = {
        "best_model": best_model_name,
        "evaluation": eval_results,
        "test_metrics": {
            "accuracy": round(accuracy_score(y_test, best_y_pred) * 100, 2),
            "macro_f1": round(f1_score(y_test, best_y_pred, average="macro") * 100, 2),
            "weighted_f1": round(f1_score(y_test, best_y_pred, average="weighted") * 100, 2),
            "confusion_matrix": cm,
            "classes": target_names
        }
    }
    with open(os.path.join(MODEL_DIR, "metrics_report.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Model artifacts successfully generated!")
    print("   -> models/sentiment_model.pkl")
    print("   -> models/word_vectorizer.pkl")
    print("   -> models/char_vectorizer.pkl")
    print("   -> models/label_encoder.pkl")
    print("   -> models/metrics_report.json")

if __name__ == "__main__":
    main()
