"""
Export trained PyTorch models to ONNX format.
Supports all 4 model architectures: CNN, CRNN, Lightweight, MiniConformer.

Usage:
    python export_to_onnx.py                    # Export the deployed model (auto-detects type)
    python export_to_onnx.py --model conformer  # Export specific architecture
    python export_to_onnx.py --model all        # Export all architectures
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path


# ============== Model 1: CNN Classifier ==============
class CNNClassifier(nn.Module):
    """Lightweight CNN for audio classification."""

    def __init__(self, n_mels=80, num_classes=150, dropout=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# ============== Model 2: CRNN Classifier ==============
class CRNNClassifier(nn.Module):
    """CRNN (CNN + BiLSTM) for audio classification."""

    def __init__(self, n_mels=80, num_classes=150, hidden_size=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.cnn_out_size = 128 * (n_mels // 8)
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.classifier(context)
        return output


# ============== Model 3: Lightweight (Depthwise Separable CNN) ==============
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightweightClassifier(nn.Module):
    """Lightweight classifier using depthwise separable convolutions."""

    def __init__(self, n_mels=80, num_classes=150, dropout=0.3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.ds_blocks = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            DepthwiseSeparableConv(128, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            DepthwiseSeparableConv(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.initial_conv(x)
        x = self.ds_blocks(x)
        x = self.classifier(x)
        return x


# ============== Model 4: MiniConformer ==============
class ConformerBlock(nn.Module):
    """Simplified Conformer block for classification."""

    def __init__(self, d_model=128, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=31, padding=15, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + self.attn_dropout(attn_out)
        conv_in = self.conv_norm(x)
        conv_in = conv_in.transpose(1, 2)
        conv_out = self.conv(conv_in)
        conv_out = conv_out.transpose(1, 2)
        x = x + conv_out
        x = x + 0.5 * self.ff2(x)
        x = self.final_norm(x)
        return x


class MiniConformerClassifier(nn.Module):
    """Mini Conformer for audio classification (encoder only)."""

    def __init__(self, n_mels=80, num_classes=150, d_model=128, n_layers=4, dropout=0.1):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        subsample_out = 32 * (n_mels // 4)
        self.proj = nn.Linear(subsample_out, d_model)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.subsample(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        x = self.proj(x)
        for block in self.conformer_blocks:
            x = block(x)
        attn_weights = F.softmax(self.attention_pool(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        output = self.classifier(x)
        return output


# ============== Model registry ==============
MODEL_REGISTRY = {
    'cnn': CNNClassifier,
    'crnn': CRNNClassifier,
    'lightweight': LightweightClassifier,
    'conformer': MiniConformerClassifier,
}


def detect_model_type(state_dict):
    """Auto-detect model type from state_dict keys."""
    keys = set(state_dict.keys())

    if any('conformer_blocks' in k for k in keys):
        return 'conformer'
    if any('lstm' in k for k in keys):
        return 'crnn'
    if any('ds_blocks' in k for k in keys):
        return 'lightweight'
    if any('conv_layers' in k for k in keys):
        return 'cnn'

    return None


def export_single_model(model_type, model_path, mapping_path, output_path, n_mels=80):
    """Export a single model to ONNX."""
    # Load class mapping
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    num_classes = len(mapping['idx_to_class'])
    print(f"\nExporting '{model_type}' model ({num_classes} classes)")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('model_config', {})
    else:
        state_dict = checkpoint
        config = {}

    # Auto-detect model type if needed
    actual_type = model_type
    if model_type == 'auto':
        detected = detect_model_type(state_dict)
        if detected:
            actual_type = detected
            print(f"  Auto-detected model type: {actual_type}")
        else:
            # Fallback: try from config
            actual_type = config.get('model_type', 'conformer')
            print(f"  Using config model_type: {actual_type}")

    # Create model
    ModelClass = MODEL_REGISTRY.get(actual_type)
    if not ModelClass:
        print(f"  ERROR: Unknown model type '{actual_type}'")
        return False

    model = ModelClass(n_mels=n_mels, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input
    dummy_input = torch.randn(1, n_mels, 300)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['mel_spectrogram'],
        output_names=['logits'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size', 2: 'time_frames'},
            'logits': {0: 'batch_size'}
        },
        dynamo=False
    )

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported to: {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    return True


def export_all_models():
    """Export the deployed model. If --model all, try loading each architecture."""
    parser = argparse.ArgumentParser(description='Export audio classifier to ONNX')
    parser.add_argument('--model', type=str, default='auto',
                        choices=['auto', 'cnn', 'crnn', 'lightweight', 'conformer', 'all'],
                        help='Model architecture to export (default: auto-detect)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (default: trained/audio_classifier_deployment.pth)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    model_path = Path(args.checkpoint) if args.checkpoint else base_dir / "trained" / "audio_classifier_deployment.pth"
    mapping_path = base_dir / "trained" / "class_mapping.json"

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    if not mapping_path.exists():
        print(f"ERROR: Class mapping not found: {mapping_path}")
        return

    print("=" * 60)
    print("Audio Classifier ONNX Export")
    print("=" * 60)

    if args.model == 'all':
        # Try each model type - only succeeds if weights match
        for model_type in MODEL_REGISTRY:
            output_path = base_dir / "trained" / f"audio_classifier_{model_type}.onnx"
            try:
                export_single_model(model_type, model_path, mapping_path, output_path)
            except Exception as e:
                print(f"\n  Skipping '{model_type}': weights don't match ({e})")
    else:
        output_path = base_dir / "trained" / "audio_classifier.onnx"
        export_single_model(args.model, model_path, mapping_path, output_path)

    print("\nDone!")


if __name__ == "__main__":
    export_all_models()
