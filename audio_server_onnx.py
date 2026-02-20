"""
Real-time Audio Classification Server using ONNX Runtime (No PyTorch required)
With LQC Code System integration for defect logging.

Requirements:
    pip install fastapi uvicorn onnxruntime numpy scipy pandas openpyxl

Usage:
    1. First run: python export_to_onnx.py (to create the ONNX model)
    2. Then run: python audio_server_onnx.py
    3. Open http://localhost:8000 in browser
"""

import json
import numpy as np
from pathlib import Path
from scipy import signal
from datetime import datetime
import pandas as pd

try:
    import onnxruntime as ort
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError as e:
    print("Please install required packages:")
    print("pip install fastapi uvicorn onnxruntime numpy scipy pandas openpyxl")
    print(f"Missing: {e}")
    exit(1)


# ============== LQC Code System ==============
class LQCCodeSystem:
    """Manages the LQC Code System hierarchy from Excel."""

    def __init__(self, excel_path):
        self.df = pd.read_excel(excel_path, sheet_name='Sheet1')
        self.df = self.df.fillna('')

        # Build hierarchy lookups
        self._build_lookups()

    def _build_lookups(self):
        """Build lookup dictionaries for quick access."""
        # Level 1: Main categories
        self.lvl1 = {}
        for _, row in self.df[['LVL1 Code', 'LVL1 Name']].drop_duplicates().iterrows():
            if row['LVL1 Name']:
                name = str(row['LVL1 Name']).strip().lower()
                self.lvl1[name] = row['LVL1 Code']

        # Level 2: Parts/Components (grouped by LVL1)
        self.lvl2 = {}
        self.lvl2_by_parent = {}
        self.lvl2_aliases = {}  # Map simplified names to full names

        for _, row in self.df[['LVL1 Code', 'LVL2 Code', 'LVL2 Name']].drop_duplicates().iterrows():
            if row['LVL2 Name']:
                name = str(row['LVL2 Name']).strip().lower()
                self.lvl2[name] = {
                    'code': row['LVL2 Code'],
                    'parent': row['LVL1 Code'],
                    'display_name': row['LVL2 Name']
                }

                # Create aliases for easier matching
                # e.g., "cap deco(f)" -> also match "cap_decof", "cap decof"
                simplified = name.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_')
                self.lvl2_aliases[simplified] = name
                # Also add without underscores
                no_special = ''.join(c for c in name if c.isalnum() or c == ' ').strip()
                self.lvl2_aliases[no_special] = name
                # Add first word as alias if multi-word
                words = name.split()
                if len(words) > 0:
                    self.lvl2_aliases[words[0]] = name

                # Group by parent
                parent = row['LVL1 Code']
                if parent not in self.lvl2_by_parent:
                    self.lvl2_by_parent[parent] = []
                self.lvl2_by_parent[parent].append({
                    'name': row['LVL2 Name'],
                    'code': row['LVL2 Code']
                })

        # Level 3: Defects (grouped by LVL2)
        self.lvl3 = {}
        self.lvl3_by_parent = {}
        self.lvl3_aliases = {}  # Map simplified names to full names

        for _, row in self.df[['LVL2 Code', 'LVL3 Code', 'LVL3 Name']].drop_duplicates().iterrows():
            if row['LVL3 Name']:
                name = str(row['LVL3 Name']).strip().lower()
                self.lvl3[name] = {
                    'code': row['LVL3 Code'],
                    'parent': row['LVL2 Code'],
                    'display_name': row['LVL3 Name']
                }

                # Create aliases for easier matching
                # e.g., "crack/tear" -> also match "crack", "tear"
                # e.g., "coming off/not inserting" -> also match "coming", "off", "inserting"
                parts = name.replace('/', ' ').replace('_', ' ').split()
                for part in parts:
                    clean_part = ''.join(c for c in part if c.isalnum()).lower()
                    if clean_part and len(clean_part) > 1:
                        self.lvl3_aliases[clean_part] = name

                # Also add simplified version
                simplified = name.replace('/', '_').replace(' ', '_')
                self.lvl3_aliases[simplified] = name

                # Group by parent
                parent = row['LVL2 Code']
                if parent not in self.lvl3_by_parent:
                    self.lvl3_by_parent[parent] = []
                self.lvl3_by_parent[parent].append({
                    'name': row['LVL3 Name'],
                    'code': row['LVL3 Code']
                })

        # Full code lookup - by LVL2 code + LVL3 code for easier matching
        self.full_codes = {}
        self.code_by_lvl2_lvl3 = {}  # (lvl2_code, lvl3_code) -> full_code

        for _, row in self.df.iterrows():
            if row['Full Code'] and row['LVL2 Code'] and row['LVL3 Code']:
                lvl2_code = str(row['LVL2 Code']).strip()
                lvl3_code = str(row['LVL3 Code']).strip()

                self.code_by_lvl2_lvl3[(lvl2_code, lvl3_code)] = {
                    'code': row['Full Code'],
                    'description': row.get('Description', ''),
                    'lvl2_name': row['LVL2 Name'],
                    'lvl3_name': row['LVL3 Name']
                }

                # Also store by names
                key = (
                    str(row['LVL2 Name']).strip().lower() if row['LVL2 Name'] else '',
                    str(row['LVL3 Name']).strip().lower() if row['LVL3 Name'] else ''
                )
                self.full_codes[key] = {
                    'code': row['Full Code'],
                    'description': row.get('Description', '')
                }

        print(f"Loaded LQC System: {len(self.lvl1)} LVL1, {len(self.lvl2)} LVL2, {len(self.lvl3)} LVL3, {len(self.code_by_lvl2_lvl3)} codes")

    def get_lvl1_names(self):
        """Get all Level 1 category names."""
        return list(set(self.lvl1.keys()))

    def get_lvl2_for_lvl1(self, lvl1_name):
        """Get Level 2 options for a given Level 1."""
        lvl1_code = self.lvl1.get(lvl1_name.lower())
        if lvl1_code and lvl1_code in self.lvl2_by_parent:
            return self.lvl2_by_parent[lvl1_code]
        return []

    def get_lvl3_for_lvl2(self, lvl2_name):
        """Get Level 3 defects for a given Level 2."""
        lvl2_info = self.lvl2.get(lvl2_name.lower())
        if lvl2_info and lvl2_info['code'] in self.lvl3_by_parent:
            return self.lvl3_by_parent[lvl2_info['code']]
        return []

    def get_full_code(self, lvl2_name, lvl3_name):
        """Get the full code for a LVL2 + LVL3 combination."""
        if not lvl2_name or not lvl3_name:
            return None

        lvl2_lower = lvl2_name.lower().strip()
        lvl3_lower = lvl3_name.lower().strip()

        # Try by name first (exact match)
        key = (lvl2_lower, lvl3_lower)
        if key in self.full_codes:
            return self.full_codes[key]

        # Find lvl2 info (might be stored by display name or simplified name)
        lvl2_info = None
        if lvl2_lower in self.lvl2:
            lvl2_info = self.lvl2[lvl2_lower]
        else:
            # Search by display name
            for name, info in self.lvl2.items():
                if info.get('display_name', '').lower() == lvl2_lower:
                    lvl2_info = info
                    break

        if not lvl2_info:
            return None

        lvl2_code = lvl2_info['code']

        # Now find matching lvl3 for this lvl2
        lvl3_clean = ''.join(c for c in lvl3_lower if c.isalnum())

        for (l2c, l3c), code_info in self.code_by_lvl2_lvl3.items():
            if l2c == lvl2_code:
                entry_lvl3_name = code_info.get('lvl3_name', '').lower()
                entry_clean = ''.join(c for c in entry_lvl3_name if c.isalnum())

                # Exact match
                if entry_lvl3_name == lvl3_lower:
                    return code_info

                # Partial match - lvl3 word appears in entry
                if lvl3_clean and (lvl3_clean in entry_clean or entry_clean.startswith(lvl3_clean)):
                    return code_info

                # Check if any word from lvl3_name matches
                lvl3_words = lvl3_lower.replace('/', ' ').replace('_', ' ').split()
                for word in lvl3_words:
                    word_clean = ''.join(c for c in word if c.isalnum())
                    if word_clean and word_clean in entry_clean:
                        return code_info

        return None

    def find_word_level(self, word):
        """Find which level a word belongs to with fuzzy matching."""
        word_lower = word.lower().strip().replace('_', ' ').replace('-', ' ')
        word_clean = ''.join(c for c in word_lower if c.isalnum()).lower()

        # Check Level 1
        if word_lower in self.lvl1:
            return {'level': 1, 'name': word, 'code': self.lvl1[word_lower]}

        # Check Level 2 - exact match first
        if word_lower in self.lvl2:
            info = self.lvl2[word_lower]
            return {'level': 2, 'name': info.get('display_name', word), **info}

        # Check Level 2 - alias match
        if word_clean in self.lvl2_aliases:
            full_name = self.lvl2_aliases[word_clean]
            info = self.lvl2[full_name]
            return {'level': 2, 'name': info.get('display_name', full_name), **info}

        # Check Level 2 - partial match (word appears in any LVL2 name)
        for name, info in self.lvl2.items():
            if word_clean in name.replace(' ', '').replace('/', '').replace('(', '').replace(')', ''):
                return {'level': 2, 'name': info.get('display_name', name), **info}

        # Check Level 3 - exact match first
        if word_lower in self.lvl3:
            info = self.lvl3[word_lower]
            return {'level': 3, 'name': info.get('display_name', word), **info}

        # Check Level 3 - alias match
        if word_clean in self.lvl3_aliases:
            full_name = self.lvl3_aliases[word_clean]
            info = self.lvl3[full_name]
            return {'level': 3, 'name': info.get('display_name', full_name), **info}

        # Check Level 3 - partial match
        for name, info in self.lvl3.items():
            name_clean = name.replace(' ', '').replace('/', '').replace('(', '').replace(')', '')
            if word_clean in name_clean or name_clean.startswith(word_clean):
                return {'level': 3, 'name': info.get('display_name', name), **info}

        return None


# ============== Mel Spectrogram Extractor ==============
class MelSpectrogramExtractor:
    """Extract mel-spectrogram features using only NumPy and SciPy."""

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160,
                 win_length=400, n_mels=80, max_frames=300):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.mel_filterbank = self._create_mel_filterbank()

    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _create_mel_filterbank(self):
        low_freq = 0
        high_freq = self.sample_rate / 2
        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        n_fft_bins = self.n_fft // 2 + 1
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        filterbank = np.zeros((self.n_mels, n_fft_bins))
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)
        return filterbank

    def extract(self, audio):
        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.float32)
        audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        window = signal.windows.hann(self.win_length, sym=False)
        if self.win_length < self.n_fft:
            pad_left = (self.n_fft - self.win_length) // 2
            pad_right = self.n_fft - self.win_length - pad_left
            window = np.pad(window, (pad_left, pad_right))

        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.n_fft) // self.hop_length
        if n_frames <= 0:
            audio = np.pad(audio, (0, self.n_fft - n_samples + self.hop_length))
            n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length

        stft = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            windowed = frame * window
            spectrum = np.fft.rfft(windowed)
            stft[:, i] = spectrum

        power_spec = np.abs(stft) ** 2
        mel_spec = np.dot(self.mel_filterbank, power_spec)
        log_mel = np.log(mel_spec + 1e-9)

        if log_mel.shape[1] > self.max_frames:
            log_mel = log_mel[:, :self.max_frames]
        elif log_mel.shape[1] < self.max_frames:
            padding = self.max_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, padding)))

        return log_mel.astype(np.float32)


# ============== ONNX Classifier ==============
class ONNXAudioClassifier:
    """Audio classifier using ONNX Runtime."""

    def __init__(self, model_path, class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            mapping = json.load(f)
        self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
        self.num_classes = len(self.idx_to_class)

        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.feature_extractor = MelSpectrogramExtractor()
        print(f"ONNX model loaded with {self.num_classes} classes")

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def classify(self, audio_data, sample_rate=16000, top_k=5):
        if isinstance(audio_data, bytes):
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = signal.resample(audio_data, new_length)

        features = self.feature_extractor.extract(audio_data)
        features = features[np.newaxis, :, :]
        logits = self.session.run([self.output_name], {self.input_name: features})[0]
        probs = self.softmax(logits)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append({
                'class': self.idx_to_class[int(idx)],
                'probability': float(probs[idx])
            })
        return predictions


# ============== Defect Logger ==============
class DefectLogger:
    """Logs defect codes to file."""

    def __init__(self, log_file='defect_log.json'):
        self.log_file = Path(log_file)
        self.logs = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)

    def log(self, entry):
        entry['timestamp'] = datetime.now().isoformat()
        self.logs.append(entry)
        self._save()
        return entry

    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def get_logs(self):
        return self.logs

    def clear(self):
        self.logs = []
        self._save()


# ============== FastAPI Server ==============
app = FastAPI(title="Audio Classification Server with LQC System")

# Add CORS middleware to allow browser WebSocket connections from any origin
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
classifier = None
lqc_system = None
defect_logger = None


@app.on_event("startup")
async def startup_event():
    global classifier, lqc_system, defect_logger
    base_dir = Path(__file__).parent

    # Load ONNX model
    model_path = base_dir / "trained" / "best_model.onnx"
    mapping_path = base_dir / "trained" / "class_mapping.json"
    if model_path.exists() and mapping_path.exists():
        classifier = ONNXAudioClassifier(str(model_path), str(mapping_path))
    else:
        print(f"Warning: Model not found at {model_path}")

    # Load LQC Code System
    excel_path = base_dir / "LQC Code System.xlsx"
    if excel_path.exists():
        lqc_system = LQCCodeSystem(str(excel_path))
    else:
        print(f"Warning: LQC Code System not found at {excel_path}")

    # Initialize defect logger
    defect_logger = DefectLogger(base_dir / "defect_log.json")


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content, headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })
    return HTMLResponse("<h1>Audio Classification Server</h1>")


@app.get("/wstest")
async def wstest():
    html_path = Path(__file__).parent / "ws_test.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Test file not found</h1>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "lqc_loaded": lqc_system is not None,
        "num_classes": classifier.num_classes if classifier else 0
    }


@app.get("/api/lqc/lvl1")
async def get_lvl1():
    """Get all Level 1 categories."""
    if not lqc_system:
        return JSONResponse({"error": "LQC system not loaded"}, status_code=500)
    return {"categories": lqc_system.get_lvl1_names()}


@app.get("/api/lqc/lvl2/{lvl1_name}")
async def get_lvl2(lvl1_name: str):
    """Get Level 2 options for a Level 1 category."""
    if not lqc_system:
        return JSONResponse({"error": "LQC system not loaded"}, status_code=500)
    return {"parts": lqc_system.get_lvl2_for_lvl1(lvl1_name)}


@app.get("/api/lqc/lvl3/{lvl2_name}")
async def get_lvl3(lvl2_name: str):
    """Get Level 3 defects for a Level 2 part."""
    if not lqc_system:
        return JSONResponse({"error": "LQC system not loaded"}, status_code=500)
    return {"defects": lqc_system.get_lvl3_for_lvl2(lvl2_name)}


@app.get("/api/logs")
async def get_logs():
    """Get all defect logs."""
    if not defect_logger:
        return JSONResponse({"error": "Logger not initialized"}, status_code=500)
    return {"logs": defect_logger.get_logs()}


@app.delete("/api/logs")
async def clear_logs():
    """Clear all defect logs."""
    if defect_logger:
        defect_logger.clear()
    return {"status": "cleared"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")

    # Session state for building defect code
    session_state = {
        'lvl1': None,
        'lvl2': None,
        'lvl3': None,
        'lvl4': None
    }

    try:
        while True:
            data = await websocket.receive_bytes()

            if classifier is None:
                await websocket.send_json({"error": "Model not loaded"})
                continue

            try:
                audio_array = np.frombuffer(data, dtype=np.float32)
                rms = np.sqrt(np.mean(audio_array ** 2))

                if rms < 0.01:
                    await websocket.send_json({"status": "silence", "rms": float(rms)})
                    continue

                # Classify
                predictions = classifier.classify(audio_array, sample_rate=16000, top_k=5)
                top_word = predictions[0]['class'] if predictions else None

                # Check if word matches LQC system
                lqc_match = None
                next_options = None
                full_code = None

                if lqc_system and top_word:
                    lqc_match = lqc_system.find_word_level(top_word)

                    if lqc_match:
                        level = lqc_match['level']

                        if level == 1:
                            session_state = {'lvl1': lqc_match.get('name', top_word), 'lvl2': None, 'lvl3': None, 'lvl4': None}
                            next_options = {
                                'level': 2,
                                'prompt': f"Select Part/Component for {lqc_match.get('name', top_word)}:",
                                'options': lqc_system.get_lvl2_for_lvl1(top_word)
                            }

                        elif level == 2:
                            display_name = lqc_match.get('display_name', lqc_match.get('name', top_word))
                            session_state['lvl2'] = display_name
                            session_state['lvl3'] = None  # Reset lvl3 when new lvl2 selected
                            next_options = {
                                'level': 3,
                                'prompt': f"Select Defect for {display_name}:",
                                'options': lqc_system.get_lvl3_for_lvl2(display_name)
                            }

                        elif level == 3:
                            session_state['lvl3'] = lqc_match.get('display_name', top_word)
                            # Try to get full code
                            if session_state['lvl2']:
                                code_info = lqc_system.get_full_code(
                                    session_state['lvl2'],
                                    session_state['lvl3']
                                )
                                if code_info:
                                    full_code = code_info
                                    # Log the defect
                                    if defect_logger:
                                        defect_logger.log({
                                            'lvl2': session_state['lvl2'],
                                            'lvl3': session_state['lvl3'],
                                            'code': code_info['code'],
                                            'description': code_info.get('description', '')
                                        })

                response = {
                    "status": "success",
                    "predictions": predictions,
                    "rms": float(rms),
                    "lqc_match": lqc_match,
                    "session_state": session_state,
                    "next_options": next_options,
                    "full_code": full_code
                }

                await websocket.send_json(response)

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        print("WebSocket client disconnected")


if __name__ == "__main__":
    print("=" * 60)
    print("Audio Classification Server with LQC Code System")
    print("=" * 60)
    print("\nStarting server at http://localhost:8001")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002)
