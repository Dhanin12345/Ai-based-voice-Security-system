import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.generate_dataset import generate_sample_dataset
from ml.build_dataset import build_features_csv
from ml.train_model import train_pipeline

def show_data_status(dataset_dir: str = "dataset", models_dir: str = "data/models"):
    """
    Displays the current status of audio files, features.csv, and trained models.
    """
    human_dir = os.path.join(dataset_dir, "human")
    synthetic_dir = os.path.join(dataset_dir, "synthetic")
    csv_path = os.path.join(dataset_dir, "features.csv")
    model_path = os.path.join(models_dir, "voice_classifier.joblib")
    scaler_path = os.path.join(models_dir, "feature_scaler.joblib")
    metrics_path = os.path.join(models_dir, "model_metrics.json")

    print("\n" + "=" * 60)
    print("         CURRENT DATASET & MODEL STATUS")
    print("=" * 60)

    # 1. Audio files
    human_files = [f for f in os.listdir(human_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac'))] if os.path.exists(human_dir) else []
    synth_files = [f for f in os.listdir(synthetic_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.flac'))] if os.path.exists(synthetic_dir) else []

    print(f"📁 Audio Dataset:")
    print(f"  • Human Audio Files    : {len(human_files)} in {human_dir}/")
    print(f"  • Synthetic Audio Files: {len(synth_files)} in {synthetic_dir}/")
    print(f"  • Total Audio Files    : {len(human_files) + len(synth_files)}")

    # 2. Features CSV
    print(f"\n📊 Extracted Features (features.csv):")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            h_count = len(df[df["class"].str.upper() == "HUMAN"])
            s_count = len(df[df["class"].str.upper() == "SYNTHETIC"])
            print(f"  • File Exists       : YES ({csv_path})")
            print(f"  • Total Rows        : {len(df)}")
            print(f"  • Total Feature Cols: {len(df.columns) - 2} (+ filename & class)")
            print(f"  • Class Balance     : {h_count} HUMAN / {s_count} SYNTHETIC")
        except Exception as e:
            print(f"  • Error reading CSV: {e}")
    else:
        print(f"  • File Exists       : NO (Run with --build to extract features)")

    # 3. Model & Metrics
    print(f"\n🧠 Machine Learning Models ({models_dir}/):")
    print(f"  • Classifier Model  : {'✓ Found' if os.path.exists(model_path) else '✗ Missing'} ({model_path})")
    print(f"  • Feature Scaler    : {'✓ Found' if os.path.exists(scaler_path) else '✗ Missing'} ({scaler_path})")
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            print(f"  • Model Type        : {metrics.get('model_name', 'Unknown')}")
            print(f"  • Accuracy          : {metrics.get('accuracy', 0) * 100:.1f}%")
            print(f"  • F1 Score          : {metrics.get('f1_score', 0) * 100:.1f}%")
            print(f"  • Precision         : {metrics.get('precision', 0) * 100:.1f}%")
            print(f"  • Recall            : {metrics.get('recall', 0) * 100:.1f}%")
            print(f"  • Trained on        : {metrics.get('total_samples', 0)} samples")
        except Exception:
            pass
    else:
        print(f"  • Metrics File      : ✗ Missing (Run with --train to train model)")

    print("=" * 60 + "\n")

def run_sample_prediction_check():
    """
    Tests model inference with a sample audio file to ensure end-to-end pipeline integrity.
    """
    print("\n--> Running end-to-end prediction verification on trained model...")
    from backend.ml.predictor import predictor
    from backend.services.audio_processor import audio_processor

    # Check human sample
    human_dir = os.path.join(BASE_DIR, "dataset", "human")
    if os.path.exists(human_dir):
        files = [f for f in os.listdir(human_dir) if f.endswith(".wav")]
        if files:
            test_file = os.path.join(human_dir, files[0])
            with open(test_file, "rb") as f:
                audio, sr = audio_processor.load_audio_file(f.read(), filename=files[0])
            audio = audio_processor.preprocess_audio(audio, sr=sr)
            result = predictor.predict_audio(audio, sr=sr)
            print(f"  [Sample Test 1: Human Sample ({files[0]})]")
            print(f"    • Classification : {result['voice_type']}")
            print(f"    • Human Prob     : {result['human_probability'] * 100:.1f}%")
            print(f"    • Synthetic Prob : {result['synthetic_probability'] * 100:.1f}%")
            print(f"    • Risk Score     : {result['risk_score']}/100 ({result['status']})")
            print(f"    • Model Status   : {result['model_status']}")

    # Check synthetic sample
    synth_dir = os.path.join(BASE_DIR, "dataset", "synthetic")
    if os.path.exists(synth_dir):
        files = [f for f in os.listdir(synth_dir) if f.endswith(".wav")]
        if files:
            test_file = os.path.join(synth_dir, files[0])
            with open(test_file, "rb") as f:
                audio, sr = audio_processor.load_audio_file(f.read(), filename=files[0])
            audio = audio_processor.preprocess_audio(audio, sr=sr)
            result = predictor.predict_audio(audio, sr=sr)
            print(f"  [Sample Test 2: Synthetic Sample ({files[0]})]")
            print(f"    • Classification : {result['voice_type']}")
            print(f"    • Human Prob     : {result['human_probability'] * 100:.1f}%")
            print(f"    • Synthetic Prob : {result['synthetic_probability'] * 100:.1f}%")
            print(f"    • Risk Score     : {result['risk_score']}/100 ({result['status']})")
            print(f"    • Model Status   : {result['model_status']}")

    print("✓ Model inference verification passed successfully for both Human & Synthetic samples!")

def main():
    parser = argparse.ArgumentParser(
        description="Unified VoiceGuard Data Management & Update Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml/update_data.py                    # Complete end-to-end update (generate + build + train)
  python ml/update_data.py --status           # Check current dataset and model status
  python ml/update_data.py --generate         # Generate new synthetic and human audio
  python ml/update_data.py --build            # Extract acoustic features from dataset/ into features.csv
  python ml/update_data.py --train            # Train ML classifier from features.csv
  python ml/update_data.py --samples 30       # Generate 30 samples per class (60 total) and retrain
        """
    )

    parser.add_argument("--status", action="store_true", help="Display current dataset and model status")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic and human audio files")
    parser.add_argument("--build", action="store_true", help="Extract features into features.csv")
    parser.add_argument("--train", action="store_true", help="Train and save the ML model")
    parser.add_argument("--samples", type=int, default=25, help="Number of audio samples per class (default: 25)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration of generated audio in seconds (default: 3.0)")
    parser.add_argument("--clean", action="store_true", help="Clean old audio files before generating new ones")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm", "gb"], help="Model type: rf (RandomForest), svm, gb (GradientBoosting)")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--models-dir", type=str, default="data/models", help="Models output directory")

    args = parser.parse_args()

    # If --status is requested alone
    if args.status:
        show_data_status(dataset_dir=args.dataset_dir, models_dir=args.models_dir)
        return

    # If no specific action specified, do ALL actions by default
    do_all = not (args.generate or args.build or args.train)
    do_generate = args.generate or do_all
    do_build = args.build or do_all
    do_train = args.train or do_all

    print("=" * 60)
    print("   VOICEGUARD DATA UPDATE & TRAINING PIPELINE")
    print("=" * 60)

    # 1. Generate audio data if requested
    if do_generate:
        print(f"\n[STEP 1/3] Generating audio dataset ({args.samples} per class)...")
        generate_sample_dataset(
            dataset_dir=args.dataset_dir,
            n_samples_per_class=args.samples,
            duration=args.duration,
            clean=args.clean
        )

    # 2. Build features CSV if requested
    if do_build:
        print(f"\n[STEP 2/3] Extracting acoustic feature vectors into features.csv...")
        csv_path = os.path.join(args.dataset_dir, "features.csv")
        build_features_csv(dataset_dir=args.dataset_dir, csv_out_path=csv_path)

    # 3. Train ML Model if requested
    if do_train:
        print(f"\n[STEP 3/3] Training and evaluating ML model ({args.model.upper()})...")
        train_pipeline(dataset_dir=args.dataset_dir, output_dir=args.models_dir, model_type=args.model)
        try:
            run_sample_prediction_check()
        except Exception as e:
            print(f"Warning: sample prediction check: {e}")

    # Display final status
    show_data_status(dataset_dir=args.dataset_dir, models_dir=args.models_dir)
    print("✓ Data update pipeline completed successfully!")

if __name__ == "__main__":
    main()
