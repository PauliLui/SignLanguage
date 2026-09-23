#!/bin/bash
# Schnelle Einrichtung des Sign Language Recognition Projekts

echo "=========================================="
echo "Sign Language Recognition - Setup"
echo "=========================================="

# 1. Python-Version prüfen
echo ""
echo "1. Python-Version prüfen..."
python3 --version

# 2. Überprüfe ob pip installiert ist
if ! command -v pip &> /dev/null; then
    echo "❌ pip nicht gefunden!"
    exit 1
fi
echo "✓ pip gefunden"

# 3. Erstelle Virtual Environment (optional aber empfohlen)
echo ""
echo "2. Virtual Environment wird erstellt..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual Environment erstellt"
else
    echo "✓ Virtual Environment existiert bereits"
fi

# 4. Aktiviere Virtual Environment
echo ""
echo "3. Virtual Environment wird aktiviert..."
source venv/bin/activate
echo "✓ Virtual Environment aktiviert"

# 5. Upgrade pip
echo ""
echo "4. pip wird aktualisiert..."
pip install --upgrade pip

# 6. Installiere Requirements
echo ""
echo "5. Dependencies werden installiert..."
pip install -r requirements.txt
echo "✓ Dependencies installiert"

# 7. Erstelle data/raw Verzeichnis
echo ""
echo "6. Datensatz-Verzeichnis wird erstellt..."
mkdir -p data/raw/{A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}
echo "✓ Verzeichnisse erstellt: data/raw/A-Z"

echo ""
echo "=========================================="
echo "✓ Setup abgeschlossen!"
echo "=========================================="
echo ""
echo "Nächste Schritte:"
echo "1. Bilder in data/raw/A, data/raw/B, etc. einfügen"
echo "2. Datensatz aufteilen: python3 dataset_manager.py"
echo "3. Trainieren: python3 train.py"
echo "4. Vorhersagen: python3 predict.py --help"
echo ""
echo "Tipp: Virtual Environment aktivieren mit: source venv/bin/activate"
echo "=========================================="
