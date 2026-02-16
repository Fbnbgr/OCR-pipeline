# OCR Pipeline - Produktionssystem

Automatisierte OCR-Verarbeitung großer PDF-Dokumente mit Tesseract und OCRmyPDF.

## Schnellstart

```bash
# 1. Virtuelle Umgebung aktivieren
source .venv/bin/activate

# 2. Setup starten
setup.bat bzw. setup.sh

# 3. PDFs in input/ Ordner kopieren
cp /path/to/pdfs/*.pdf input/

# 4. Pipeline ausführen
python ocr_process.py
```
### Windows
choco install ghostscript -y


## Was macht die Pipeline?

1. **Requirements Installation**: Installiert alle Python-Abhängigkeiten
2. **OCR Verarbeitung**: Extrahiert Text aus PDF-Bildern mit Tesseract (Deutsch+Englisch)
3. **Text-Layer Hinzufügen**: Erstellt durchsuchbare PDFs mit ocrmypdf
4. **Logging & Statistiken**: Detailliertes Logging mit Gesamtstatistiken

## Verzeichnisstruktur

```
OCR pipeline/
├── ocr_process.py          ← HAUPTDATEI
├── requirements.txt         ← Dependencies
├── input/                  ← PDF-Eingabe
├── output/                 ← Text-Extrakte (.txt)
├── output_with_text_layer/ ← Finale PDFs mit Textebene
├── logs/                   ← Protokolldateien
└── .venv/                  ← Python Virtual Environment
```

## Output

Nach der Verarbeitung finden Sie:

- **`output/*.txt`**: Extrahierter Text für jedes Dokument
- **`output_with_text_layer/*.pdf`**: Finale durchsuchbare PDFs
- **`output/ocr_statistics.json`**: Detaillierte Statistiken (Zeichen, Wörter, Seiten)
- **`logs/ocr_pipeline.log`**: Vollständiges Verarbeitungslog

## Konfiguration

Systemanforderungen:
- Tesseract OCR (installiert und im PATH)
- Python 3.10+
- Mindestens 4GB RAM (für PDF-Verarbeitung)

Unterstützte Sprachen:
- Deutsch (deu)
- Englisch (eng)

## Performance

- DPI: 100 (optimiert für große Dateien)
- Durchsatz: ~20-30 Seiten pro Minute
- Speichernutzung: Effizient durch Streaming

## Logging

Die Konsolenausgabe zeigt:
- Verarbeitungsfortschritt pro Datei
- Extrahierte Zeichen- und Wortanzahlen
- Größen der Ein-/Ausgabedateien

Die Log-Datei enthält zusätzlich:
- DEBUG-Informationen
- Detaillierte Fehlerberichte
- Komplette Verarbeitungshistorie

## Anforderungen

Alle benötigten Packages sind in `requirements.txt` aufgelistet:
- **pytesseract**: Python-Interface zu Tesseract
- **pdf2image**: PDF zu Bild-Konvertierung
- **ocrmypdf**: OCR und Textebenen-Hinzufügung
- **PyMuPDF**: PDF-Manipulation
