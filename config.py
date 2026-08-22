"""
Konfigurationsdatei für Sign Language Recognition Projekt
"""
import os
from pathlib import Path

# Pfade
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Datensatz-Konfiguration
DATASET_CONFIG = {
    "train_split": 0.7,      # 70% Training
    "val_split": 0.15,       # 15% Validierung
    "test_split": 0.15,      # 15% Testing
    "random_seed": 42,
    "image_size": (224, 224),
    "batch_size": 32,
}

# Modell-Konfiguration
MODEL_CONFIG = {
    "num_classes": 29,  # A-Z + del + nothing + space
    "input_shape": (224, 224, 3),
    "dropout_rate": 0.5,
    "learning_rate": 0.001,
}

# Training-Konfiguration
TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

# MediaPipe Konfiguration
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# Alphabet A-Z + Sonderzeichen
ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['del', 'nothing', 'space']
ALPHABET_MAPPING = {char: idx for idx, char in enumerate(ALPHABET)}
REVERSE_ALPHABET_MAPPING = {idx: char for char, idx in ALPHABET_MAPPING.items()}

# Datensatz-Struktur
# data/raw/
#   ├── A/
#   ├── B/
#   ├── C/
#   └── ... (A-Z)
#
# Die Bilder werden automatisch in train/val/test aufgeteilt

print("✓ Konfiguration geladen")
print(f"✓ Projekt-Root: {PROJECT_ROOT}")
print(f"✓ Daten-Verzeichnis: {DATA_DIR}")
print(f"✓ Anzahl Klassen: {MODEL_CONFIG['num_classes']} (A-Z + del + nothing + space)")
