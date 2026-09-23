# 📊 PROJEKT-ÜBERSICHT

## ✅ Was wurde erstellt?

Ein **komplettes Sign Language Recognition System** mit Python 3.11, TensorFlow und MediaPipe.

### Ordnerstruktur:
```
/home/pauli/projects/SignLanguage/
├── data/
│   ├── raw/         ← Deine Trainingsdaten (Bilder der Buchstaben A-Z)
│   ├── train/       ← Auto-generiert: 70% der Bilder
│   ├── val/         ← Auto-generiert: 15% der Bilder  
│   ├── test/        ← Auto-generiert: 15% der Bilder
│   └── processed/   ← Für Augmentation
├── models/          ← Trainierte Modelle (.h5)
├── logs/            ← TensorBoard Logs
├── output/          ← Vorhersage-Ergebnisse
│
├── config.py        ← Konfiguration (alle Parameter)
├── dataset_manager.py ← Datensatz laden + aufteilen
├── model.py         ← TensorFlow CNN + MediaPipe
├── train.py         ← Trainings-Skript
├── predict.py       ← Vorhersage-Skript
├── augment.py       ← Daten-Augmentation
├── utils.py         ← Analyse-Tools
│
├── setup.sh         ← Setup-Skript (automatic)
├── requirements.txt ← Dependencies
├── README.md        ← Vollständige Doku
└── QUICKSTART.md    ← Quick Start Guide
```

---

## 🎯 Die 5 Hauptschritte

### **1. Bilder vorbereiten** 📸
- Sammle Fotos von Gebärdensprache-Buchstaben (A-Z)
- Strukturiert nach Ordnern: `data/raw/A/`, `data/raw/B/`, ..., `data/raw/Z/`

### **2. Dependencies installieren** 📦
```bash
cd /home/pauli/projects/SignLanguage
pip install -r requirements.txt
```

### **3. Datensatz aufteilen** ✂️
```bash
python3 dataset_manager.py
```
→ Teilt automatisch in 70% Train, 15% Val, 15% Test

### **4. Modell trainieren** 🎓
```bash
python3 train.py
```
→ Trainiert ein tiefes CNN mit 50 Epochen

### **5. Vorhersagen machen** 🔮
```bash
python3 predict.py --image data/test/A/image.jpg
python3 predict.py --webcam
python3 predict.py --video video.mp4
```

---

## 🏗️ Modell-Details

**Architektur:**
- 4x Convolutional Blocks (32→256 Filter)
- Batch Normalization
- MaxPooling + Dropout
- Global Average Pooling
- 2x Fully Connected Layers (512→256)
- Output: 26 Klassen (A-Z)

**Parameter:** ~2.5 Million

**Framework:**
- TensorFlow/Keras für Deep Learning
- MediaPipe für Hand-Detection
- OpenCV für Bildverarbeitung

---

## 📚 Python Module

**config.py** - Alle Einstellungen:
```python
- Datensatz-Split (70/15/15)
- Bild-Größe (224x224)
- Batch Size (32)
- Learning Rate (0.001)
- MediaPipe Parameter
- Alphabet-Mapping (A-Z)
```

**dataset_manager.py** - Datensatz-Management:
```python
- load_dataset_from_raw()    - Lade Bilder
- _split_and_copy_files()    - Teile in Train/Val/Test
- load_images_from_split()   - Lade aus Split
- print_dataset_info()       - Zeige Statistik
```

**model.py** - TensorFlow Modell:
```python
- build_model()              - Baue CNN
- extract_hand_landmarks()   - MediaPipe Integration
- predict()                  - Einzelne Vorhersagen
- predict_batch()            - Batch Vorhersagen
```

**train.py** - Training:
```python
- prepare_data()             - Laden & normalisieren
- setup_callbacks()          - Early Stopping, Checkpoints
- train()                    - Trainiere Modell
- evaluate()                 - Teste auf Test-Set
```

**predict.py** - Vorhersagen:
```python
- predict_image()            - Einzelbild
- predict_batch()            - Mehrere Bilder
- predict_video()            - Video/Webcam
```

**augment.py** - Daten-Augmentation:
```python
- rotate(), flip(), scale()  - Transformationen
- brightness(), contrast()   - Farb-Anpassungen
- augment_dataset()          - Augmentiere Datensatz
```

**utils.py** - Analyse:
```python
- analyze_dataset()          - Datensatz-Statistik
- check_corrupted_images()   - QA
- display_sample_images()    - Beispiel-Anzeige
```

---

## 📋 requirements.txt

```
tensorflow==2.15.0       # Deep Learning
mediapipe==0.10.5        # Hand Detection
numpy==1.24.3            # Numerisch
opencv-python==4.8.1.78  # Bildverarbeitung
pillow==10.0.0           # Bildformate
scikit-learn==1.3.1      # ML Tools
matplotlib==3.8.0        # Visualisierung
tqdm==4.66.1             # Progress Bars
pyyaml==6.0.1            # Config Files
```

---

## 🚀 Schnelle Befehle

```bash
# Setup
bash setup.sh

# Analyse
python3 utils.py

# Datensatz aufteilen
python3 dataset_manager.py

# Trainieren
python3 train.py

# Vorhersagen
python3 predict.py --image <path>
python3 predict.py --webcam
python3 predict.py --batch <dir>

# Augmentation
python3 augment.py

# TensorBoard
tensorboard --logdir=logs
```

---

## 📊 Expected Results

Mit 100 Bildern pro Buchstabe (2.600 Gesamtbilder):

| Metrik | Wert |
|--------|------|
| Training Accuracy | 85-95% |
| Validation Accuracy | 75-85% |
| Test Accuracy | 70-80% |
| Training Zeit | 5-30 Min |

Mit 200+ Bildern pro Buchstabe:
- +5-10% Accuracy
- +5-10 Min Training Time

---

## 🔧 Konfigurierbare Parameter

**In `config.py` ändern:**

```python
# Datensatz
DATASET_CONFIG['train_split'] = 0.7
DATASET_CONFIG['image_size'] = (224, 224)
DATASET_CONFIG['batch_size'] = 32

# Modell
MODEL_CONFIG['learning_rate'] = 0.001
MODEL_CONFIG['dropout_rate'] = 0.5

# Training
TRAINING_CONFIG['epochs'] = 50
TRAINING_CONFIG['early_stopping_patience'] = 10

# MediaPipe
MEDIAPIPE_CONFIG['min_detection_confidence'] = 0.5
```

---

## 💡 Tipps für beste Ergebnisse

✅ **Datensatz:**
- Min. 50-100 Bilder pro Buchstabe
- Gute Beleuchtung
- Deutlich erkennbare Handhaltung
- Verschiedene Winkel & Hintergründe

✅ **Training:**
- Mehr Epochen (50-100)
- Größere Batch Size (32-64)
- Data Augmentation verwenden
- Validierungs-Accuracy monitoren

✅ **Vorhersage:**
- Med-Konfidenz Threshold setzen
- Hand-Landmarks für Debugging anschauen
- Mehrere Frames bei Video averaging

---

## 🎓 Projektpfad

```
📍 Location: /home/pauli/projects/SignLanguage
📌 Python Version: 3.11+ (empfohlen)
📌 Framework: TensorFlow 2.15+
📌 ML Libraries: MediaPipe, Keras, OpenCV, scikit-learn
```

---

## 📝 Was ist zu tun?

1. ✅ **Setup abgeschlossen** - Alle Dateien erstellt ✓
2. ⏳ **TODO:** Bilder in `data/raw/A-Z/` einfügen
3. ⏳ **TODO:** `python3 dataset_manager.py` ausführen
4. ⏳ **TODO:** `python3 train.py` starten
5. ⏳ **TODO:** `python3 predict.py --webcam` testen

---

## 📖 Weitere Ressourcen

- [TensorFlow Docs](https://www.tensorflow.org)
- [MediaPipe Docs](https://mediapipe.dev)
- [README.md](README.md) - Vollständige Dokumentation
- [QUICKSTART.md](QUICKSTART.md) - Quick Start Guide

---

**🎉 Projekt fertig zur Verwendung!**

Lade deine Bilder hoch und starte das Training! 🚀
