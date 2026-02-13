@echo off
REM OCR Pipeline Setup Script für Windows

echo ========================================
echo OCR Pipeline - Setup für Windows
echo ========================================

REM Prüfe ob Virtual Environment existiert
if not exist ".venv" (
    echo Erstelle Virtual Environment...
    python -m venv .venv
)

REM Aktiviere Virtual Environment
echo Aktiviere Virtual Environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrade pip...
python -m pip install --upgrade pip

REM Installiere Requirements
echo Installiere Requirements...
pip install -r requirements.txt

REM Erstelle Verzeichnisse
echo.
echo Erstelle Verzeichnisse...
mkdir input 2>nul
mkdir output 2>nul
mkdir logs 2>nul

echo.
echo ========================================
echo Setup abgeschlossen! [OK]
echo ========================================
echo.
echo Nächste Schritte:
echo 1. Lege Input-Dateien in 'input\' ab
echo 2. Führe aus: python ocr_pipeline.py
echo.
echo Weitere Informationen siehe README.md
echo.
pause
