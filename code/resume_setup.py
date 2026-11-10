"""
resume_setup.py — Indian LPR CRNN setup module (local Windows version)
"""
import os
import sys
import importlib.util
import torch
import cv2
import numpy as np

# --- Paths ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')

# --- Load CRNN class from local file ---
_CRNN_PY = os.path.join(_SCRIPT_DIR, 'crnn_model.py')
_spec = importlib.util.spec_from_file_location("crnn_module", _CRNN_PY)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['crnn_module'] = _mod
_spec.loader.exec_module(_mod)
CRNN = _mod.CRNN

# --- Config ---
CHARSET     = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUM_CLASSES = 37
INPUT_H     = 32
INPUT_W     = 160
PTH = os.path.join(WEIGHTS_DIR, 'crnn_tiny_plate.pth')

if not os.path.exists(PTH):
    raise FileNotFoundError(f"CRNN weights not found: {PTH}")

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(in_channel=3, num_classes=NUM_CLASSES, cnn_input_height=INPUT_H,
             is_tiny=True, use_gru=True).to(device)

_state = torch.load(PTH, map_location=device, weights_only=True)
# Some CRNN saves wrap the state_dict under 'state_dict' key
if isinstance(_state, dict) and 'state_dict' in _state:
    _state = _state['state_dict']

model.load_state_dict(_state, strict=True)
model.eval()

print(f"✅ CRNN loaded ({sum(p.numel() for p in model.parameters()):,} params on {device})")

# --- Preprocess ---
def preprocess(img_bgr):
    if img_bgr is None:
        raise ValueError("preprocess received None — image load failed")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img_bgr, (INPUT_W, INPUT_H))
    data = torch.from_numpy(img).float() / 255.0
    return data.permute(2, 0, 1).unsqueeze(0)

# --- CTC decode ---
def decode(log_probs):
    if log_probs.dim() == 3 and log_probs.size(0) == 1:
        log_probs = log_probs.squeeze(0)
    elif log_probs.dim() == 3 and log_probs.size(1) == 1:
        log_probs = log_probs.squeeze(1)
    _, idxs = log_probs.max(-1)
    text, prev = [], -1
    for i in idxs.tolist():
        if i != prev and i != 0:
            text.append(CHARSET[i - 1])
        prev = i
    return ''.join(text)

# --- Post-processing rules for Indian plates ---
def fix_indian_plate(text):
    if len(text) < 4:
        return text

    # First 2 characters must be letters (state code)
    chars = list(text)
    for i in range(min(2, len(chars))):
        if chars[i] == '0': chars[i] = 'O'
        elif chars[i] == '1': chars[i] = 'I'
        elif chars[i] == '5': chars[i] = 'S'
        elif chars[i] == '8': chars[i] = 'B'

    text = ''.join(chars)

    # Common state-code correction
    state_fixes = {
        'OB': 'PB', '0B': 'PB',   # Punjab misread
        '0H': 'MH', 'OH': 'MH',   # Maharashtra misread
        'KN': 'KA',               # Karnataka misread
    }
    if len(text) >= 2:
        prefix = text[:2]
        if prefix in state_fixes:
            text = state_fixes[prefix] + text[2:]

    return text


@torch.no_grad()
def recognize(image_bgr):
    if image_bgr is None:
        raise ValueError("recognize received None — image load failed")
    raw = decode(model(preprocess(image_bgr).to(device)))
    return fix_indian_plate(raw)