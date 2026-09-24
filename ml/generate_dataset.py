import os
import io
import argparse
import numpy as np
import soundfile as sf

def generate_sample_dataset(
    dataset_dir: str = "dataset",
    n_samples_per_class: int = 25,
    duration: float = 3.0,
    sr: int = 16000,
    clean: bool = False
):
    """
    Generates realistic audio training data (.wav) for HUMAN and SYNTHETIC speech classes:
      - dataset/human/    : Authentic speech characteristics (pitch modulation, formants, pauses, micro-jitter)
      - dataset/synthetic/: Synthetic/AI voice artifacts (vocoder buzz, frozen F0, spectral hiss, metallic artifacts)
    """
    human_dir = os.path.join(dataset_dir, "human")
    synthetic_dir = os.path.join(dataset_dir, "synthetic")

    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)

    if clean:
        for d in [human_dir, synthetic_dir]:
            for f in os.listdir(d):
                if f.endswith(".wav"):
                    try:
                        os.remove(os.path.join(d, f))
                    except Exception:
                        pass
        print(f"Cleaned existing .wav files in '{human_dir}' and '{synthetic_dir}'.")

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    print(f"--> Generating {n_samples_per_class} HUMAN samples and {n_samples_per_class} SYNTHETIC samples ({duration}s each @ {sr}Hz)...")

    # 1. Generate HUMAN Voice Samples
    for i in range(1, n_samples_per_class + 1):
        voice_type = i % 3  # 0: male register, 1: female register, 2: expressive conversational
        if voice_type == 0:
            base_f0 = np.random.uniform(95.0, 145.0)    # Male fundamental
        elif voice_type == 1:
            base_f0 = np.random.uniform(175.0, 240.0)   # Female fundamental
        else:
            base_f0 = np.random.uniform(130.0, 200.0)   # Expressive / mid

        # Dynamic natural pitch contour with vibrato + conversational phrase inflection + natural micro-jitter
        vibrato_rate = np.random.uniform(4.0, 6.5)
        intonation_slope = np.random.uniform(-35.0, 35.0) * np.sin(2 * np.pi * 0.8 * t)
        vibrato = 22.0 * np.sin(2 * np.pi * vibrato_rate * t)
        jitter = np.random.normal(0, 2.5, size=len(t)) # vocal fold micro-tremor
        f0_contour = np.clip(base_f0 + intonation_slope + vibrato + jitter, 70.0, 420.0)

        phase = 2 * np.pi * np.cumsum(f0_contour) / sr

        # Human vocal cord harmonic structure with formant resonances
        harmonics = (
            0.45 * np.sin(phase) +
            0.25 * np.sin(2 * phase) +
            0.15 * np.sin(3 * phase) +
            0.08 * np.sin(4 * phase) +
            0.04 * np.sin(5 * phase)
        )

        # Formant resonant enhancement (simulate oral vowel cavity F1 ~ 600-800Hz)
        formant_freq = np.random.uniform(500.0, 900.0)
        formant_wave = 0.12 * np.sin(2 * np.pi * formant_freq * t)

        # Natural syllabic envelope modulation with natural breath pause intervals
        syllable_rate = np.random.uniform(2.5, 4.5)
        syllables = np.clip(np.sin(2 * np.pi * syllable_rate * t) + 0.3, 0.0, 1.0)
        
        # Soft attack / decay at edges
        fade_len = int(sr * 0.05)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        envelope = np.ones(len(t), dtype=np.float32)
        envelope[:fade_len] = fade_in
        envelope[-fade_len:] = fade_out
        envelope = envelope * (0.35 + 0.65 * syllables)

        speech = ((harmonics + formant_wave) * envelope).astype(np.float32)
        # Subtle acoustic background noise
        speech += np.random.normal(0, 0.003, size=len(speech)).astype(np.float32)
        # Peak normalization
        max_val = np.max(np.abs(speech))
        if max_val > 0:
            speech = speech / max_val * 0.85

        file_path = os.path.join(human_dir, f"human_{i:03d}.wav")
        sf.write(file_path, speech, sr)

    # 2. Generate SYNTHETIC Voice Samples (AI voice cloning & vocoder artifacts)
    for i in range(1, n_samples_per_class + 1):
        artifact_type = i % 4
        
        if artifact_type == 0:
            # Artifact 0: Rigid Pitch Freeze (zero pitch modulation / flat robotic contour)
            fixed_f0 = np.random.uniform(110.0, 180.0)
            f0_contour = fixed_f0 * np.ones_like(t)
            phase = 2 * np.pi * np.cumsum(f0_contour) / sr
            # Vocoder buzzy harmonics
            speech = (0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase) + 0.1 * np.sin(4 * phase))
            # Vocoder white noise leakage
            speech += 0.035 * np.random.randn(len(t))

        elif artifact_type == 1:
            # Artifact 1: Neural Vocoder High-Frequency Flatness & Phase Discontinuity
            f0 = np.random.uniform(120.0, 160.0) + 1.5 * np.sin(2 * np.pi * 1.0 * t) # very flat modulation
            phase = 2 * np.pi * np.cumsum(f0) / sr
            speech = 0.4 * np.sin(phase) + 0.25 * np.sin(2 * phase)
            # High spectral flatness artifact (high-frequency hiss in 3kHz - 8kHz band)
            high_freq_noise = np.random.normal(0, 0.045, size=len(t))
            speech = speech + high_freq_noise

        elif artifact_type == 2:
            # Artifact 2: Quantized Step-Pitch (staircase pitch jumps, typical of TTS vocoders)
            f0_steps = np.repeat(np.random.choice([120.0, 140.0, 125.0, 150.0], 8), len(t) // 8 + 1)[:len(t)]
            phase = 2 * np.pi * np.cumsum(f0_steps) / sr
            speech = 0.45 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
            # Metallic ring modulation
            speech = speech * (1.0 + 0.15 * np.sin(2 * np.pi * 320.0 * t))

        else:
            # Artifact 3: Over-smoothed envelope & metallic harmonics
            f0 = 135.0 * np.ones_like(t)
            phase = 2 * np.pi * np.cumsum(f0) / sr
            speech = 0.4 * np.sin(phase) + 0.3 * np.cos(2.05 * phase) + 0.2 * np.sin(3.02 * phase)
            # Unnatural flat envelope without breath pauses
            speech = speech * 0.75 + 0.02 * np.random.randn(len(t))

        # Peak normalization
        max_val = np.max(np.abs(speech))
        if max_val > 0:
            speech = (speech / max_val * 0.85).astype(np.float32)

        file_path = os.path.join(synthetic_dir, f"synthetic_{i:03d}.wav")
        sf.write(file_path, speech, sr)

    total_files = n_samples_per_class * 2
    print(f"✓ Successfully generated {total_files} audio files in '{dataset_dir}/' ({n_samples_per_class} human, {n_samples_per_class} synthetic).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic and human voice training dataset")
    parser.add_argument("--samples", type=int, default=25, help="Number of samples per class (default: 25)")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration of each sample in seconds (default: 3.0)")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate (default: 16000)")
    parser.add_argument("--clean", action="store_true", help="Remove old .wav files before generating")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Target dataset directory")

    args = parser.parse_args()
    generate_sample_dataset(
        dataset_dir=args.dataset_dir,
        n_samples_per_class=args.samples,
        duration=args.duration,
        sr=args.sr,
        clean=args.clean
    )
