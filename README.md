# GMES LQC Defect Logging System

A real-time audio-based defect classification and logging system for LG manufacturing quality control. Operators speak defect descriptions, and the system classifies audio into parts and defect codes using deep learning, then logs them with hierarchical LQC codes.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Audio Classification Models](#audio-classification-models)
- [LQC Code System](#lqc-code-system)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Browser Compatibility](#browser-compatibility)

## Overview

The system has two modes of operation:

1. **Standalone Mode** - Open `index.html` directly in a browser. The full UI works offline with simulated audio classification for demos and training.

2. **Server Mode** - Run the FastAPI backend for real-time audio classification via WebSocket. Audio is captured from the microphone, sent to the server, classified by a trained ONNX model, and matched to LQC defect codes.

## Architecture

```
Browser (index.html)
    |
    |-- WebSocket (ws://localhost:8000/ws)
    |
FastAPI Server (audio_server_onnx.py)
    |
    |-- ONNX Runtime (audio_classifier.onnx)
    |-- LQC Code System (LQC Code System.xlsx)
    |-- Defect Logger (defect_log.json)
```

- **Frontend**: Single-file HTML5 application with vanilla CSS and JavaScript. No frameworks or build tools.
- **Backend**: Python FastAPI server with ONNX Runtime inference. Serves the frontend and handles WebSocket audio streaming.
- **ML Pipeline**: PyTorch models trained on labeled audio data, exported to ONNX for production inference.

## Project Structure

```
LG-2/
├── index.html                  # Main application UI (standalone + server mode)
├── audio_server_onnx.py        # FastAPI backend with ONNX inference
├── export_to_onnx.py           # PyTorch to ONNX model export utility
├── start_classifier.py         # Simple HTTP server for static file serving
├── organize_audio.py           # Audio data organization utility
├── requirements.txt            # Python dependencies
├── LQC Code System.xlsx        # Hierarchical defect code definitions
├── defect_log.json             # Logged defect history
├── trained/
│   ├── audio_classifier.onnx           # Production ONNX model (4.8 MB)
│   ├── audio_classifier_deployment.pth # PyTorch checkpoint
│   ├── best_model.pth                  # Training best model
│   └── class_mapping.json              # 150 class label mappings
├── audio/                      # Raw audio recordings
├── audio_classified/           # Training data organized by class
└── audio_classification.ipynb  # Training notebook
```

## Setup

### Standalone Mode (No Installation)

Open `index.html` in any modern browser. No server, no dependencies, no internet required.

### Server Mode

**Requirements**: Python 3.8+

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn onnxruntime numpy scipy pandas openpyxl
   ```

2. Ensure model files exist in `trained/`:
   - `audio_classifier.onnx`
   - `class_mapping.json`

3. Start the server:
   ```bash
   python audio_server_onnx.py
   ```

4. Open http://localhost:8000 in your browser.

### Training a New Model

If you need to retrain or export a new model:

1. Organize audio data into `audio_classified/` with one folder per class.
2. Train using `audio_classification.ipynb`.
3. Export to ONNX:
   ```bash
   python export_to_onnx.py                    # Auto-detect architecture
   python export_to_onnx.py --model conformer  # Specify architecture
   python export_to_onnx.py --model all        # Export all architectures
   ```

## Usage

### Navigation Flow

1. **View Parts** - The Level 2 grid shows all available parts (Basket, Display, Door, etc.)
2. **Select Part** - Click a part to see its defects in the Level 3 grid
3. **Select Defect** - Click a defect type (Crack, Burn, Scratch, etc.)
4. **Confirm** - Click "LQC CONFIRM" to log the defect with its full code
5. **Repeat** - Continue logging defects as needed

### Voice Input (Server Mode)

1. Hold **SPACEBAR** to start recording from the microphone
2. Speak the part name or defect type
3. Release to stop recording
4. The system classifies the audio and highlights the matching grid item
5. Confirm or correct the selection

### Status Indicators

| Status       | Color  | Meaning                        |
|-------------|--------|--------------------------------|
| READY       | Green  | Waiting for input              |
| RECORDING   | Red    | Microphone active              |
| CLASSIFYING | Amber  | Processing audio               |

## Audio Classification Models

The system supports four model architectures, all trained on mel-spectrogram features (80 mel bins, 16kHz sample rate):

| Model         | Type                          | Description                                   |
|--------------|-------------------------------|-----------------------------------------------|
| CNN          | Convolutional Neural Network   | 4-layer CNN with batch norm and adaptive pooling |
| CRNN         | CNN + Bidirectional LSTM       | CNN feature extraction with LSTM sequence modeling and attention |
| Lightweight  | Depthwise Separable CNN        | Mobile-friendly architecture with reduced parameters |
| MiniConformer| Conformer (Transformer-based)  | Self-attention + convolution blocks with attention pooling |

All models classify into **150 classes** covering parts and defect types.

**Inference Pipeline**:
```
Microphone Audio (PCM float32, 16kHz)
    → Mel Spectrogram (80 bins x 300 frames)
    → ONNX Runtime Inference
    → Top-K Predictions with Confidence
    → LQC Code Matching
```

## LQC Code System

The defect code hierarchy is defined in `LQC Code System.xlsx` with three levels:

- **Level 1**: Main categories (top-level grouping)
- **Level 2**: Parts/Components (e.g., Basket, Display, Door, Hinge, Ice Maker)
- **Level 3**: Defect types (e.g., Crack/Tear, Burn, Scratch, Distance/Gap, Missing)

Each Level 2 + Level 3 combination maps to a unique **Full Code** (e.g., `MAGSA1` for Case Lamp Cover > Distance/Gap).

The system supports fuzzy matching and aliases so that spoken words are matched even with partial or alternate names.

## API Reference

### REST Endpoints

| Method | Endpoint              | Description                      |
|--------|-----------------------|----------------------------------|
| GET    | `/`                   | Serves the frontend UI           |
| GET    | `/health`             | Server health and model status   |
| GET    | `/api/lqc/lvl1`       | List Level 1 categories          |
| GET    | `/api/lqc/lvl2/{name}`| Level 2 parts for a Level 1      |
| GET    | `/api/lqc/lvl3/{name}`| Level 3 defects for a Level 2    |
| GET    | `/api/logs`           | Retrieve all defect logs         |
| DELETE | `/api/logs`           | Clear all defect logs            |

### WebSocket

**Endpoint**: `ws://localhost:8000/ws`

**Send**: Raw PCM float32 audio bytes

**Receive**: JSON response
```json
{
  "status": "success",
  "predictions": [
    {"class": "Basket", "probability": 0.85},
    {"class": "Bottom", "probability": 0.07}
  ],
  "rms": 0.045,
  "lqc_match": {"level": 2, "name": "Basket", "code": "..."},
  "session_state": {"lvl1": null, "lvl2": "Basket", "lvl3": null},
  "next_options": {"level": 3, "prompt": "Select Defect for Basket:", "options": [...]},
  "full_code": null
}
```

Silence detection (RMS < 0.01) returns `{"status": "silence"}`.

## Deployment

### Local

```bash
python audio_server_onnx.py
# Server runs at http://localhost:8000
```

### Production (Render - Free Tier)

1. Push to GitHub (ensure `trained/audio_classifier.onnx`, `trained/class_mapping.json`, and `LQC Code System.xlsx` are included)
2. Create a Web Service on [render.com](https://render.com)
3. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn audio_server_onnx:app --host 0.0.0.0 --port 8000`
4. The backend serves both the API and the frontend from a single service

### Static Frontend Only (Netlify)

For demo/offline mode without the backend:
1. Deploy `index.html` to [Netlify](https://app.netlify.com) via drag-and-drop
2. The UI works standalone with simulated classification

## Browser Compatibility

| Browser           | Status      |
|-------------------|-------------|
| Chrome / Edge     | Supported   |
| Firefox           | Supported   |
| Safari            | Supported   |
| Internet Explorer | Not supported |

Microphone access requires HTTPS in production or `localhost` during development.

## Tech Stack

| Layer      | Technology                                    |
|-----------|-----------------------------------------------|
| Frontend  | HTML5, CSS3, Vanilla JavaScript (ES6+)        |
| Backend   | Python, FastAPI, Uvicorn                      |
| ML        | PyTorch (training), ONNX Runtime (inference)  |
| Audio     | Web Audio API (browser), SciPy/NumPy (server) |
| Data      | Excel (openpyxl), JSON                        |

## License

Proprietary - Internal use only.
