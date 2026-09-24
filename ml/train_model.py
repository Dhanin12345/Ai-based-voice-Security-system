import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.audio_processor import audio_processor
from backend.services.feature_extractor import feature_extractor

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm", ".opus"}

def load_dataset_features(path: str = "dataset"):
    """
    Loads features and class labels from dataset/features.csv.
    """
    if os.path.isdir(path):
        csv_path = os.path.join(path, "features.csv")
    else:
        csv_path = path

    if not os.path.exists(csv_path):
        from ml.build_dataset import build_features_csv
        df = build_features_csv(dataset_dir=path if os.path.isdir(path) else "dataset", csv_out_path=csv_path)
    else:
        df = pd.read_csv(csv_path)

    if df is None or df.empty:
        raise ValueError(f"Features dataset '{csv_path}' is empty.")

    label_map = {"HUMAN": 0, "SYNTHETIC": 1}
    y = df["class"].map(label_map).values
    
    # Drop non-feature columns
    drop_cols = ["filename", "class"]
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    X = feature_df.values.astype(np.float32)

    return X, y, list(df["filename"])

def train_pipeline(dataset_dir: str = "dataset", output_dir: str = "data/models", model_type: str = "rf"):
    print("==================================================")
    print("  VOICEGUARD ML TRAINING & EVALUATION PIPELINE")
    print("==================================================")

    os.makedirs(output_dir, exist_ok=True)

    X, y, filenames = load_dataset_features(dataset_dir)
    print(f"Loaded {len(X)} total audio samples.")
    print(f"  - Human samples (label 0): {np.sum(y == 0)}")
    print(f"  - Synthetic samples (label 1): {np.sum(y == 1)}")

    if len(X) < 4:
        raise ValueError("Insufficient dataset samples for training. Please provide at least 4 audio files.")

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    test_size = 0.25 if len(X) >= 8 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Model Selection
    if model_type.lower() == "svm":
        model = SVC(probability=True, random_state=42)
        model_name = "Support Vector Classifier (SVM)"
    elif model_type.lower() in ["gb", "gradient_boosting"]:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_name = "Gradient Boosting Classifier"
    else:
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model_name = "Random Forest Classifier"

    print(f"\nTraining model: {model_name}...")
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n---------------- MODEL EVALUATION METRICS ----------------")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1 Score:  {f1 * 100:.2f}%")
    print(f"  Confusion Matrix:\n  {cm}")
    print("----------------------------------------------------------")

    # Save artifacts
    model_file = os.path.join(output_dir, "voice_classifier.joblib")
    scaler_file = os.path.join(output_dir, "feature_scaler.joblib")
    metrics_file = os.path.join(output_dir, "model_metrics.json")

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    metrics_data = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "total_samples": len(X),
        "test_samples": len(X_test)
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nSuccessfully saved trained model to: {model_file}")
    print(f"Successfully saved feature scaler to: {scaler_file}")
    print(f"Successfully saved evaluation metrics to: {metrics_file}")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoiceGuard ML Classifier")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="data/models", help="Output directory for saved model")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm", "gb"], help="Model architecture")

    args = parser.parse_args()
    train_pipeline(dataset_dir=args.dataset, output_dir=args.output, model_type=args.model)
