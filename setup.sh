#!/bin/bash

# OCR Pipeline Setup Script

echo "========================================"
echo "OCR Pipeline - Setup"
echo "========================================"

# Prüfe ob Virtual Environment existiert
if [ ! -d ".venv" ]; then
    echo "Erstelle Virtual Environment..."
    python3 -m venv .venv
fi

# Aktiviere Virtual Environment
echo "Aktiviere Virtual Environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrade pip..."
pip install --upgrade pip

# Installiere Requirements
echo "Installiere Requirements..."
pip install -r requirements.txt

# Prüfe auf Systemabhängigkeiten
echo ""
echo "Prüfe Systemabhängigkeiten..."

# Prüfe auf ghostscript (für OCRMYPDF)
if ! command -v gs &> /dev/null; then
    echo "⚠️  ghostscript nicht gefunden. Für OCRMYPDF ist ghostscript erforderlich:"
    echo "   Linux (Ubuntu/Debian): sudo apt-get install ghostscript"
    echo "   macOS: brew install ghostscript"
    echo "   Windows: Installiere von https://ghostscript.com/download/gsdnld.html"
else
    echo "✓ ghostscript gefunden"
fi

# Erstelle Verzeichnisse
echo ""
echo "Erstelle Verzeichnisse..."
mkdir -p input output logs

echo ""
echo "========================================"
echo "Setup abgeschlossen! ✓"
echo "========================================"
echo ""
echo "Nächste Schritte:"
echo "1. Lege Input-Dateien in 'input/' ab"
echo "2. Führe aus: python ocr_pipeline.py"
echo ""
echo "Weitere Informationen siehe README.md"
echo ""
