# AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Defensive voice-security system built with Python, FastAPI, Librosa, Scikit-learn, WebSockets, SQLite/SQLAlchemy, and a cyber-themed real-time frontend dashboard.

---

## Table of Contents
1. [Project Objective](#project-objective)
2. [Key Features](#key-features)
3. [High-Level Data Analysis Pipeline](#high-level-data-analysis-pipeline)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Running the Application](#running-the-application)
8. [Dataset & ML Data Pipeline (Updating & Running Data)](#dataset--ml-data-pipeline-updating--running-data)
9. [API Documentation](#api-documentation)
10. [AI Model Integration](#ai-model-integration)
11. [Database Schema](#database-schema)
12. [Testing](#testing)
13. [Privacy & Security](#privacy--security)
14. [Limitations & Disclaimer](#limitations--disclaimer)

---

## Project Objective
Build a comprehensive defensive voice-security system that:
1. Accepts real-time live browser microphone audio over WebSockets.
2. Accepts uploaded audio files (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`) for deep inspection.
3. Segments audio into short temporal sliding windows.
4. Extracts speech feature vectors (MFCCs, Mel Spectrograms, Spectral Centroid, Bandwidth, Rolloff, Spectral Flatness, ZCR, RMS Energy, Pitch F0).
5. Runs an AI model to estimate human vs. synthetic speech probability.
6. Computes a configurable risk score (0–100) and risk level classification (`LOW`, `MODERATE`, `HIGH`, `VERY HIGH`).
7. Analyzes temporal window trends across continuous streams.
8. Generates immediate visual, auditory, and persistent alerts when risk thresholds are exceeded.
9. Provides security recommendations (secondary channel verification, challenge-response questions, data withholding).
10. Maintains audit logs in an SQLite/PostgreSQL database.

---

## High-Level Data Analysis Pipeline

```
               AUDIO INPUT
                    │
         ┌──────────┴──────────┐
         │                     │
    Live Microphone        Audio Upload
         │                     │
         └──────────┬──────────┘
                    ↓
            AUDIO VALIDATION
                    ↓
           AUDIO PREPROCESSING
    (Mono, 16kHz, DC Offset, Peak Norm, Silence Trim)
                    ↓
            FEATURE EXTRACTION
    (MFCC, Mel Spec, Spectral Flatness, ZCR, Pitch F0)
                    ↓
            AI / ML DETECTION
         (VoiceCloneDetector Model)
                    ↓
       PROBABILITY ESTIMATION & RISK SCORE
                    ↓
            TEMPORAL ANALYSIS
       (Window trend aggregation)
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
      NORMAL               SUSPICIOUS
         │                     │
         │                SECURITY ALERT
         │                     ↓
         │              RECOMMENDATIONS
         └──────────┬──────────┘
                    ↓
             DATABASE STORAGE
                    ↓
            SECURITY DASHBOARD
```

---

## Technology Stack
- **Backend Framework**: FastAPI (Python 3.10+)
- **Audio Processing**: Librosa, SoundFile, NumPy, SciPy
- **AI/ML Engine**: Scikit-Learn, Joblib, Modular pre-trained model interface (PyTorch compatible)
- **Database**: SQLite (SQLAlchemy ORM, structured for PostgreSQL/MySQL scalability)
- **Real-Time Streaming**: WebSockets & Web Audio API (PCM 16-bit 16kHz stream)
- **Frontend UI**: HTML5, CSS3 (Dark Cyberpunk / Glassmorphism), Vanilla JS, Chart.js

---

## Project Structure

```
voice-cloning-detector/
│
├── backend/
│   ├── main.py                    # FastAPI app entry point & routes mounting
│   ├── config.py                  # Pydantic configuration & env settings
│   │
│   ├── api/
│   │   ├── routes_detection.py    # REST file upload & WebSocket /ws/detect routes
│   │   ├── routes_history.py      # Detection audit history & alerts API
│   │   └── routes_health.py       # System status & health check endpoints
│   │
│   ├── services/
│   │   ├── audio_processor.py     # Resampling, normalization, silence trim & windowing
│   │   ├── feature_extractor.py   # MFCC, Mel, Spectral Flatness, ZCR feature extractor
│   │   ├── detector.py            # Standardized VoiceCloneDetector interface
│   │   ├── risk_scorer.py         # Risk score calculator & recommendation generator
│   │   └── alert_service.py       # Alert generator & database recorder
│   │
│   ├── models/
│   │   ├── detection_result.py    # Pydantic request/response schemas
│   │   └── database.py            # SQLAlchemy database tables (DetectionHistory, Alert)
│   │
│   ├── ml/
│   │   ├── model_loader.py        # Pre-trained model loader & dev acoustic heuristic model
│   │   ├── inference.py           # Model inference engine
│   │   └── preprocessing.py       # Feature vector scaler
│   │
│   └── utils/
│       └── logger.py              # Structured privacy-aware logger
│
├── frontend/
│   ├── index.html                 # Overview landing page
│   ├── dashboard.html             # Real-time security dashboard with Chart.js & controls
│   ├── history.html               # Audit history log viewer & CSV exporter
│   ├── settings.html              # Threshold & notification settings
│   │
│   ├── css/
│   │   └── style.css              # Modern cybersecurity dark slate theme
│   │
│   └── js/
│       ├── app.js                 # Global notification toasts & Web Audio tone synthesizer
│       ├── recorder.js            # Browser Web Audio microphone PCM capture
│       ├── detector.js            # WebSocket client stream handler
│       ├── dashboard.js           # Chart.js timeline & dashboard controller
│       └── history.js             # Audit table filter, pagination & CSV export
│
├── data/
│   ├── uploads/                   # Temporary upload directory
│   └── database/                  # SQLite database location
│
├── tests/
│   ├── test_audio.py              # Preprocessing & feature extraction tests
│   ├── test_detector.py           # AI prediction & risk score unit tests
│   └── test_api.py                # FastAPI REST & WebSocket endpoint integration tests
│
├── requirements.txt
├── README.md
└── .env.example
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip & virtualenv

### 1. Clone & Environment Setup

#### On Windows:
```powershell
# Create Virtual Environment
python -m venv venv

# Activate Virtual Environment
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

#### On Linux / macOS:
```bash
# Create Virtual Environment
python3 -m venv venv

# Activate Virtual Environment
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI application server:
```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access the web pages in your browser:
- **Overview Page**: `http://localhost:8000/index.html` (or `http://localhost:8000`)
- **Real-Time Security Dashboard**: `http://localhost:8000/dashboard.html`
- **Detection History Logs**: `http://localhost:8000/history.html`
- **System Settings**: `http://localhost:8000/settings.html`
- **Interactive OpenAPI API Docs**: `http://localhost:8000/docs`

---

## Dataset & ML Data Pipeline (Updating & Running Data)

The system includes a fully automated acoustic feature extraction and machine learning training pipeline located in `ml/`.

### 1. Data Architecture

The data ecosystem consists of 3 layers:
- **Raw Audio Data** (`dataset/`):
  - `dataset/human/`: Authentic human audio files (`.wav`, `.mp3`, `.m4a`, etc.) with natural speech pitch inflection, formants, and syllable timing.
  - `dataset/synthetic/`: Synthetic, AI-cloned, vocoder, or text-to-speech audio files.
- **Extracted Feature Matrix** (`dataset/features.csv`):
  - 44-dimensional acoustic vectors per sample: MFCC 1-13 (means and standard deviations), spectral centroid, spectral bandwidth, spectral rolloff, spectral flatness, RMS energy, zero-crossing rate (ZCR), pitch F0 (mean, std, min, max, range, covariance), silence ratio, and voice activity ratio.
- **Trained Model Artifacts** (`data/models/`):
  - `voice_classifier.joblib`: Serialized Scikit-learn Classifier (Random Forest, SVM, or Gradient Boosting).
  - `feature_scaler.joblib`: Standard scaler matching the 44-dimensional feature vector.
  - `model_metrics.json`: Evaluated accuracy, precision, recall, F1 score, and confusion matrix.

---

### 2. How to Update Programming Data

You can update programming data in multiple ways:

#### Option A: One-Command Automated Update (Recommended)
Run the unified data updater to generate new audio samples, extract features, and retrain the model in one go:
```powershell
python ml/update_data.py --samples 25 --clean
```
Or via the root pipeline runner:
```powershell
python run_pipeline.py --update-data --samples 25
```

#### Option B: Add Your Own Custom Audio Data
1. Place authentic human recordings (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`) into `dataset/human/`.
2. Place cloned/deepfake voice recordings into `dataset/synthetic/`.
3. Extract acoustic features and retrain the model:
```powershell
python ml/update_data.py --build --train
```

#### Option C: Step-by-Step Manual Pipeline
If you prefer running each phase individually:

1. **Generate Synthetic & Human Speech**:
   ```powershell
   python ml/generate_dataset.py --samples 30 --duration 3.0 --clean
   ```

2. **Extract Acoustic Feature Vectors to CSV**:
   ```powershell
   python ml/build_dataset.py --dataset dataset --output dataset/features.csv
   ```

3. **Train the ML Classifier**:
   ```powershell
   python ml/train_model.py --dataset dataset --model rf
   ```
   *(Options for `--model`: `rf` for Random Forest, `svm` for Support Vector Machine, `gb` for Gradient Boosting)*

---

### 3. How to Run Data & View Results

#### 1. Check Data & Model Status
Inspect the current dataset file counts, class balance, and model metrics:
```powershell
python run_pipeline.py --status
# or:
python ml/update_data.py --status
```

#### 2. Windows 1-Click Interactive Menu
On Windows, you can double-click or run:
```powershell
.\run_data_pipeline.bat
```
This presents an interactive menu to update data, check status, run tests, or start the server.

#### 3. Run Application Server with Trained Data
Start the backend and UI:
```powershell
python run_pipeline.py --server
# or:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/dashboard.html` to test live microphone streams or upload audio files to see the trained model classify voices in real time.

#### 4. Run Automated Test Verification
Ensure all 15 audio processing, detection, and API tests pass:
```powershell
python run_pipeline.py --test
# or:
pytest tests/ -v
```

---

## API Documentation

FastAPI auto-generates complete OpenAPI documentation accessible at `/docs`.

### Key Endpoints:
1. `GET /health`
   - Returns service status, active sample rate, and configured risk thresholds.
2. `POST /api/detect/upload`
   - Accepts multipart audio files (`.wav`, `.mp3`, `.m4a`).
   - Returns classification, confidence, risk score, temporal breakdown, and security recommendations.
3. `WS /ws/detect`
   - Real-time WebSocket connection for streaming raw 16-bit PCM microphone chunks.
   - Pushes live detection JSON payloads to the dashboard.
4. `GET /api/history`
   - Paginated history logs with search (`?search=`), classification filter (`?classification=`), and sorting.
5. `GET /api/history/{id}`
   - Detailed detection report by record ID.

---

## AI Model Integration
The system implements a modular `VoiceCloneDetector` class (`backend/services/detector.py`).

### Replacing with Custom Pre-Trained Weights
To integrate a custom anti-spoofing model (such as AASIST, RawNet2, or a custom trained Scikit-learn/PyTorch classifier):
1. Train your model on an anti-spoofing dataset (e.g., ASVspoof 2019/2021).
2. Save the serialized model file as `data/models/anti_spoof_model.pkl` (or configure via `backend/config.py`).
3. The `ModelLoader` (`backend/ml/model_loader.py`) will automatically load your production model at runtime without changing the application logic.

---

## Database Schema

Database table models are defined in `backend/models/database.py`.

### `DetectionHistory` Table
- `id` (INT, Primary Key)
- `timestamp` (DATETIME)
- `source` (STRING: 'upload' / 'live_microphone')
- `filename` (STRING)
- `duration_seconds` (FLOAT)
- `classification` (STRING: 'likely_human' / 'possibly_synthetic')
- `confidence` (FLOAT: 0.0 – 1.0)
- `human_probability` (FLOAT)
- `synthetic_probability` (FLOAT)
- `risk_score` (INT: 0 – 100)
- `risk_level` (STRING: 'LOW', 'MODERATE', 'HIGH', 'VERY HIGH')
- `processing_time_ms` (FLOAT)
- `recommendation` (TEXT)

---

## Testing

Execute unit and integration tests using `pytest`:
```powershell
pytest tests/ -v
```

Tests cover:
- Audio loading, preprocessing, 16kHz resampling, and sliding window creation (`test_audio.py`).
- Acoustic feature vector extraction (`test_audio.py`).
- AI model interface, probability bounding, and risk scoring (`test_detector.py`).
- FastAPI REST upload endpoints, health checks, and WebSocket connectivity (`test_api.py`).

---

## Privacy & Security
1. **Biometric Privacy**: Raw microphone audio is processed in memory and never written to disk during live streaming sessions.
2. **Auto-Cleanup**: Uploaded temporary files are processed and immediately deleted.
3. **Structured Logging**: Log files record metadata and performance numbers only—never raw audio payloads or user credentials.

---

## Limitations & Disclaimer
- **Probabilistic AI Output**: AI voice cloning detection is inherently probabilistic. Results estimate statistical likelihood based on acoustic artifacts and must **not** be treated as absolute forensic or legal proof of impersonation.
- **Acoustic Noise Sensitivity**: Extremely noisy environments or heavy VoIP audio compression (e.g. GSM/PSTN telephony codecs) can affect feature extraction accuracy.
- **Verification Best Practice**: Always enforce secondary out-of-band verification before taking sensitive actions.
#   A i - b a s e d - v o i c e - S e c u r i t y - s y s t e m  
 