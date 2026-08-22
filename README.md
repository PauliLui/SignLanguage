# Sign Language Recognition mit TensorFlow & MediaPipe

Ein Deep Learning Projekt zur Erkennung von Gebärdensprache (ASL - American Sign Language) Buchstaben mit TensorFlow und MediaPipe.

## 📋 Projektübersicht

Dieses Programm:
- Trainiert ein tiefes CNN-Modell zur Klassifizierung von Gebärdensprache-Buchstaben (A-Z)
- Verwendet MediaPipe für Handgestenerkennung
- Teilt Datensätze automatisch in Training (70%), Validierung (15%) und Test (15%)
- Ermöglicht Vorhersagen auf Einzelbildern, Bilderserien und Video/Webcam

## 📁 Projektstruktur

```
SignLanguage/
├── data/
│   ├── raw/              # 👈 Hier Trainingsdaten einfügen!
│   │   ├── A/           # Bilder des Buchstaben A
│   │   ├── B/           # Bilder des Buchstaben B
│   │   └── ... Z/       # ... bis Z
│   ├── train/           # Automatisch generiert
│   ├── val/             # Automatisch generiert
│   └── test/            # Automatisch generiert
├── models/              # Trainierte Modelle
├── logs/                # TensorBoard Logs
├── output/              # Vorhersage-Ergebnisse
├── config.py            # Konfiguration
├── dataset_manager.py   # Datensatz-Manager
├── model.py             # TensorFlow Modell
├── train.py             # Training-Skript
├── predict.py           # Vorhersage-Skript
└── requirements.txt     # Python Dependencies
```

## 🚀 Installation

### 1. Python 3.11 (empfohlen)

Stelle sicher, dass Python 3.11 verwendet wird:

```bash
python3 --version
```

Falls nicht, wechsle zu Python 3.11:

```bash
# Zeige verfügbare Versionen
ls /usr/bin/python*

# Setze Standardversion (z.B. Python 3.11)
sudo ln -sf /usr/bin/python3.11 /usr/bin/python3
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

## 📊 Datensatz vorbereiten

### Schritt 1: Bilder in data/raw einfügen

Erstelle Ordner für jeden Buchstaben und füge Bilder ein:

```bash
mkdir -p data/raw/A data/raw/B data/raw/C ... data/raw/Z
```

Kopiere Bilder:
```bash
cp /path/to/images/A/* data/raw/A/
cp /path/to/images/B/* data/raw/B/
# usw...
```

### Schritt 2: Datensatz automatically aufteilen

```bash
python3 dataset_manager.py
```

Dies teilt den Datensatz auf:
- **70%** → Training (`data/train/`)
- **15%** → Validierung (`data/val/`)
- **15%** → Test (`data/test/`)

Jeder Ordner wird mit Buchstaben (A-Z) strukturiert.

## 🎓 Modell trainieren

```bash
python3 train.py
```

Das Training wird:
1. Bilder laden und normalisieren
2. Das CNN-Modell bauen
3. Für max. 50 Epochen trainieren
4. Best Model automatisch speichern
5. Test-Set evaluieren

**Trainings-Parameter** (in `config.py` änderbar):
- Epochs: 50
- Batch Size: 32
- Learning Rate: 0.001
- Early Stopping: nach 10 Epochen ohne Verbesserung

## 🔮 Vorhersagen machen

### Einzelnes Bild

```bash
python3 predict.py --image data/test/A/image.jpg
```

Mit Hand-Landmarks:
```bash
python3 predict.py --image data/test/A/image.jpg --landmarks
```

### Batch (ganzes Verzeichnis)

```bash
python3 predict.py --batch data/test/A
```

### Live-Webcam

```bash
python3 predict.py --webcam
```

Drücke **'q'** zum Beenden, **'s'** um Screenshot zu speichern.

### Video-Datei

```bash
python3 predict.py --video mein_video.mp4
```

## 🏗️ Modell-Architektur

Das CNN-Modell besteht aus:

- **4 Convolutional Blocks** mit:
  - Conv2D Layer (32 → 256 Filter)
  - Batch Normalization
  - Max Pooling
  - Dropout (0.25)

- **Global Average Pooling**

- **2 Fully Connected Layers**:
  - Dense (512) + Dropout
  - Dense (256) + Dropout
  - Dense (26) + Softmax (Output)

**Gesamt Parameter**: ~2.5M

## 📈 Trainings-Monitoring

TensorBoard Logs anschauen:

```bash
tensorboard --logdir=logs
```

Öffne http://localhost:6006 im Browser.

## 🔧 Konfiguration

Alle Einstellungen in `config.py`:

```python
# Datensatz
DATASET_CONFIG = {
    "train_split": 0.7,      # 70% Training
    "val_split": 0.15,       # 15% Validierung
    "test_split": 0.15,      # 15% Test
    "image_size": (224, 224),
    "batch_size": 32,
}

# Modell
MODEL_CONFIG = {
    "num_classes": 26,        # A-Z
    "learning_rate": 0.001,
}

# Training
TRAINING_CONFIG = {
    "epochs": 50,
    "early_stopping_patience": 10,
}

# MediaPipe
MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}
```

## 📝 Datensatz-Anforderungen

Für beste Ergebnisse:

- **Mindestens 50-100 Bilder pro Buchstabe**
- Bildformat: JPG, PNG, BMP
- Auflösung: mind. 200x200 Pixel
- Helle, gut beleuchtete Fotos
- Deutlich sichtbare Handhaltung
- Hintergrund sollte relativ konstant sein

Beispiel-Datensatz-Struktur:
```
data/raw/
├── A/
│   ├── A_001.jpg
│   ├── A_002.jpg
│   └── ... (100+ Bilder)
├── B/
│   └── ... (100+ Bilder)
└── ... Z/
```

## 🎯 Expected Performance

Mit ~100 Bildern pro Klasse:

- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 75-85%
- **Test Accuracy**: 70-80%

Performance hängt stark von Datensatz-Qualität ab.

## 🐛 Troubleshooting

**Problem: "Keine Daten gefunden"**
- Stelle sicher, dass `data/raw/A`, `data/raw/B`, etc. existieren
- Bilder müssen in diesen Ordnern sein

**Problem: Out of Memory**
- Reduziere `batch_size` in `config.py`
- Reduziere `image_size` (z.B. auf 160x160)

**Problem: Niedrige Genauigkeit**
- Mehr Trainingsdaten sammeln
- Augmentation aktivieren
- Längeres Training
- Model-Architektur verbessern

**Problem: Webcam funktioniert nicht**
```bash
# Linux: OpenCV Kamera-Zugriff
sudo usermod -a -G video $USER
```

## 📚 Dependencies

- **tensorflow** - Deep Learning Framework
- **mediapipe** - Hand Detection
- **opencv-python** - Bildverarbeitung
- **numpy** - Numerische Operationen
- **scikit-learn** - Daten-Splitting
- **pillow** - Bildformate

## 🤝 Verbesserungsmöglichkeiten

- [ ] Data Augmentation (Rotationen, Zoom, etc.)
- [ ] Transfer Learning (Pre-trained Models)
- [ ] Ensemble-Modelle
- [ ] GPU-Beschleunigung
- [ ] Mobile-Optimierung
- [ ] Real-time Performance-Optimierung

## 📄 Lizenz

MIT License

## 👨‍💻 Autor

Erstellt für Sign Language Recognition mit Deep Learning

---

**Viel Erfolg beim Training! 🚀**