import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.audio_processor import audio_processor
from backend.services.feature_extractor import feature_extractor

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm", ".opus"}

def build_features_csv(dataset_dir: str = "dataset", csv_out_path: str = "dataset/features.csv", verbose: bool = True):
    """
    Extracts acoustic feature matrices from dataset/human/ and dataset/synthetic/ and saves dataset/features.csv.
    Processes all supported audio formats (.wav, .mp3, .ogg, .flac, .m4a, .aac, .webm, .opus).
    """
    human_dir = os.path.join(dataset_dir, "human")
    synthetic_dir = os.path.join(dataset_dir, "synthetic")

    rows = []

    print(f"--> Extracting acoustic features from '{human_dir}' and '{synthetic_dir}'...")

    def process_dir(dir_path: str, class_label: str):
        if not os.path.exists(dir_path):
            print(f"Warning: Directory '{dir_path}' does not exist.")
            return

        file_list = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        total = len(file_list)
        print(f"  Found {total} audio files for class [{class_label}]. Processing...")

        for idx, fname in enumerate(file_list, 1):
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "rb") as f:
                    content = f.read()
                audio, sr = audio_processor.load_audio_file(content, filename=fname)
                audio = audio_processor.preprocess_audio(audio, sr=sr)
                
                if len(audio) > 512:
                    fdict = feature_extractor.extract_features(audio, sr=sr)
                    
                    row = {
                        "filename": fname,
                        "class": class_label,
                        "duration": round(fdict.get("duration", float(len(audio)/sr)), 4),
                        "pitch_mean": round(fdict.get("pitch_mean", 0.0), 2),
                        "pitch_std": round(fdict.get("pitch_std", 0.0), 2),
                        "pitch_min": round(fdict.get("pitch_min", 0.0), 2),
                        "pitch_max": round(fdict.get("pitch_max", 0.0), 2),
                        "pitch_range": round(fdict.get("pitch_range", 0.0), 2),
                        "rms_mean": round(fdict.get("energy_mean", 0.0), 6),
                        "rms_std": round(fdict.get("energy_std", 0.0), 6),
                        "zcr_mean": round(fdict.get("zcr_mean", 0.0), 6),
                        "zcr_std": round(fdict.get("zcr_std", 0.0), 6),
                        "spectral_centroid_mean": round(fdict.get("spectral_centroid_mean", 0.0), 2),
                        "spectral_centroid_std": round(fdict.get("spectral_centroid_std", 0.0), 2),
                        "spectral_bandwidth_mean": round(fdict.get("spectral_bandwidth_mean", 0.0), 2),
                        "spectral_bandwidth_std": round(fdict.get("spectral_bandwidth_std", 0.0), 2),
                        "spectral_rolloff_mean": round(fdict.get("spectral_rolloff_mean", 0.0), 2),
                        "spectral_rolloff_std": round(fdict.get("spectral_rolloff_std", 0.0), 2),
                        "silence_ratio": round(fdict.get("silence_ratio", 0.0), 4),
                        "voice_activity_ratio": round(fdict.get("voice_activity_ratio", 0.0), 4)
                    }

                    # Add MFCC 1..13 mean and std
                    mfcc_means = fdict.get("mfcc_mean", np.zeros(13))
                    mfcc_stds = fdict.get("mfcc_std", np.zeros(13))

                    for i in range(13):
                        row[f"mfcc_{i+1}_mean"] = round(float(mfcc_means[i]) if i < len(mfcc_means) else 0.0, 4)
                        row[f"mfcc_{i+1}_std"] = round(float(mfcc_stds[i]) if i < len(mfcc_stds) else 0.0, 4)

                    rows.append(row)

                    if verbose and (idx % 10 == 0 or idx == total):
                        print(f"    [{class_label}] Processed {idx}/{total} files ({fname})")

            except Exception as e:
                print(f"  Error processing {fname}: {e}")

    process_dir(human_dir, "HUMAN")
    process_dir(synthetic_dir, "SYNTHETIC")

    if not rows:
        print("No valid audio files processed.")
        return None

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_out_path), exist_ok=True)
    df.to_csv(csv_out_path, index=False)
    print(f"✓ Successfully generated '{csv_out_path}' with {len(df)} feature rows ({len(df.columns)} columns).")
    print(f"  - HUMAN samples: {len(df[df['class'] == 'HUMAN'])}")
    print(f"  - SYNTHETIC samples: {len(df[df['class'] == 'SYNTHETIC'])}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract acoustic features from dataset audio into CSV")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="dataset/features.csv", help="Output path for features.csv")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-batch output")

    args = parser.parse_args()
    build_features_csv(dataset_dir=args.dataset, csv_out_path=args.output, verbose=not args.quiet)
