# 🚀 QUICK START GUIDE - Sign Language Recognition

**Projektordner:** `/home/pauli/projects/SignLanguage`

## 📋 Was wurde erstellt?

✅ Komplette Python 3.11 Projektstruktur
✅ TensorFlow CNN-Modell mit MediaPipe Integration
✅ Automatische Datensatz-Aufteilung (Train/Val/Test)
✅ Training-, Vorhersage- und Augmentation-Skripte
✅ Konfigurierbare Parameter

---

## 🎯 SCHRITT-FÜR-SCHRITT ANLEITUNG

### **Schritt 1️⃣: Trainingsdaten vorbereiten**

Deine Bilder müssen in dieser Struktur sein:

```
data/raw/
├── A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (min. 50 Bilder)
├── B/
│   └── ... (50+ Bilder)
└── Z/
    └── ... (50+ Bilder)
```

**Ordner erstellen:**
```bash
cd /home/pauli/projects/SignLanguage
mkdir -p data/raw/{A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}
```

**Bilder einfügen:**
- Kopiere alle Bilder des Buchstabens A nach `data/raw/A/`
- Kopiere alle Bilder des Buchstabens B nach `data/raw/B/`
- Usw. bis Z

---

### **Schritt 2️⃣: Dependencies installieren**

```bash
cd /home/pauli/projects/SignLanguage

# Virtual Environment erstellen (optional aber empfohlen)
python3 -m venv venv
source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt
```

**Oder einfach das Setup-Skript ausführen:**
```bash
bash setup.sh
```

---

### **Schritt 3️⃣: Datensatz analysieren und aufteilen**

```bash
# Analysiere Datensatz
python3 utils.py

# Teile Datensatz auf (70% Train, 15% Val, 15% Test)
python3 dataset_manager.py
```

**Ausgabe sollte sein:**
```
Training:   500 Bilder (70%)
Validierung: 150 Bilder (15%)
Test:       150 Bilder (15%)
```

---

### **Schritt 4️⃣: Modell trainieren** 🎓

```bash
python3 train.py
```

Dies wird:
- ✓ Bilder laden und normalisieren
- ✓ CNN-Modell bauen (~2.5M Parameter)
- ✓ Training für max. 50 Epochen starten
- ✓ Best Model speichern
- ✓ Test-Set evaluieren

**Trainings-Zeit:** ~5-30 Minuten (abhängig von Datengröße & Hardware)

---

### **Schritt 5️⃣: Vorhersagen machen** 🔮

**Einzelnes Bild:**
```bash
python3 predict.py --image data/test/A/image.jpg
```

**Mehrere Bilder (Batch):**
```bash
python3 predict.py --batch data/test/A
```

**Live-Webcam:**
```bash
python3 predict.py --webcam
```
Drücke `q` zum Beenden, `s` um Screenshot zu speichern

**Video-Datei:**
```bash
python3 predict.py --video mein_video.mp4
```

---

## 📊 Datensatz-Anforderungen

Für **gute Ergebnisse** brauchst du:

- **Pro Buchstabe:** 50-100 Bilder (minimum)
- **Bildformat:** JPG, PNG
- **Auflösung:** mind. 200x200 Pixel
- **Qualität:** Gute Beleuchtung, deutlich erkennbare Handhaltung
- **Hintergrund:** Relativ gleichmäßig

**Mit mehr Daten (200+ Bilder/Buchstabe):**
- Accuracy: 85-95% (Training)
- Accuracy: 75-85% (Validierung)
- Accuracy: 70-80% (Test)

---

## 🛠️ Alle Python-Skripte

| Skript | Zweck |
|--------|-------|
| `config.py` | Konfiguration (ändern um Parameter zu setzen) |
| `dataset_manager.py` | Datensatz laden & aufteilen |
| `model.py` | TensorFlow CNN + MediaPipe |
| `train.py` | Modell trainieren |
| `predict.py` | Vorhersagen machen |
| `augment.py` | Daten-Augmentation (mehr Trainingsbilder) |
| `utils.py` | Datensatz-Analyse |

---

## 🎯 Erwartete Ordnerstruktur nach Setup

```
SignLanguage/
├── data/
│   ├── raw/          ← DEINE BILDER HIER!
│   ├── train/        ← Auto generiert (70%)
│   ├── val/          ← Auto generiert (15%)
│   └── test/         ← Auto generiert (15%)
├── models/           ← Trainierte Modelle
├── logs/             ← TensorBoard Logs
├── output/           ← Ergebnisse
├── *.py              ← Python Skripte
└── requirements.txt  ← Dependencies
```

---

## 🚨 Häufige Fehler

**❌ "Keine Daten gefunden"**
→ Stelle sicher, dass Bilder in `data/raw/A/`, `data/raw/B/`, etc. sind

**❌ "Out of Memory"**
→ Reduziere `batch_size` oder `image_size` in `config.py`

**❌ "Webcam funktioniert nicht"**
→ Linux: `sudo usermod -a -G video $USER` und neu anmelden

**❌ "Niedrige Genauigkeit"**
→ Mehr Trainingsdaten sammeln, Augmentation nutzen

---

## 💡 Tipps & Tricks

**Daten augmentieren** (wenn wenig Trainingsdaten):
```bash
python3 augment.py
```

**TensorBoard anschauen:**
```bash
tensorboard --logdir=logs
# Öffne http://localhost:6006
```

**Trainings-Parameter ändern** (in `config.py`):
```python
TRAINING_CONFIG = {
    "epochs": 100,              # Mehr Epochen
    "batch_size": 16,           # Kleinere Batches
    ...
}
```

**Modell-Architektur verbessern:**
- Editiere `model.py` und verändere `build_model()`
- Füge mehr Conv-Layer hinzu
- Verändere Filter-Größen

---

## 📞 Nächste Schritte

1. ✅ Bilder in `data/raw/A-Z/` einfügen
2. ✅ `python3 dataset_manager.py` ausführen
3. ✅ `python3 train.py` starten
4. ✅ `python3 predict.py --webcam` testen

---

**Happy Training! 🎉**

Bei Fragen: Schau in die Konfiguration in `config.py` oder README.md
