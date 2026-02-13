#!/usr/bin/env python3
"""
OCR Pipeline - Zentrale Produktionsdatei
==========================================

Durchführung:
1. Installation aller Requirements
2. Verarbeitung aller PDFs aus dem input/ Ordner mit Tesseract OCR
3. Hinzufügen von Textebenen mit ocrmypdf
4. Speicherung der Ergebnisse in output/

Logging:
- Konsolenausgabe mit detaillierten Informationen
- Datei-Logging in logs/ocr_pipeline.log mit Gesamtstatistiken
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict
import concurrent.futures

# PDF-Verarbeitung
import pytesseract
from pdf2image import convert_from_path
import ocrmypdf

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Konfiguriert Logging für Konsole und Datei."""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'ocr_pipeline.log'
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (DEBUG level)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file

logger, log_file = setup_logging()
logger.info("=" * 80)
logger.info("OCR PIPELINE GESTARTET")
logger.info("=" * 80)

# ============================================================================
# KONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_WITH_TEXT_DIR = BASE_DIR / 'output_with_text_layer'
VENV_DIR = BASE_DIR / '.venv'

# Erstelle Verzeichnisse
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_WITH_TEXT_DIR.mkdir(exist_ok=True)

logger.info(f"Arbeitsverzeichnis: {BASE_DIR}")
logger.info(f"Input: {INPUT_DIR}")
logger.info(f"Output: {OUTPUT_DIR}")

# ============================================================================
# REQUIREMENTS INSTALLATION
# ============================================================================

def install_requirements():
    """Installiert alle Python-Abhängigkeiten."""
    logger.info("\n" + "=" * 80)
    logger.info("1. INSTALLATION VON REQUIREMENTS")
    logger.info("=" * 80)
    
    requirements_file = BASE_DIR / 'requirements.txt'
    
    if not requirements_file.exists():
        logger.warning(f"requirements.txt nicht gefunden: {requirements_file}")
        return False
    
    try:
        logger.info(f"Installiere Packages aus: {requirements_file}")
        
        # Nutze den Python aus der venv
        python_exe = VENV_DIR / 'bin' / 'python'
        pip_exe = VENV_DIR / 'bin' / 'pip'
        
        if not pip_exe.exists():
            logger.error(f"pip nicht gefunden: {pip_exe}")
            return False
        
        result = subprocess.run(
            [str(pip_exe), 'install', '-q', '-r', str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.warning(f"pip install hatte Warnungen: {result.stderr}")
        else:
            logger.info("✓ Alle Requirements erfolgreich installiert")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Fehler bei Requirements Installation: {e}")
        return False


# ============================================================================
# OCR VERARBEITUNG
# ============================================================================

def process_pdf_with_tesseract(pdf_path):
    """
    Verarbeitet eine PDF mit Tesseract OCR.
    
    Returns:
        tuple: (text, char_count, word_count, page_count)
    """
    try:
        logger.info(f"  Konvertiere PDF zu Bildern...")
        images = convert_from_path(str(pdf_path), dpi=100)
        page_count = len(images)
        
        logger.info(f"  {page_count} Seiten gefunden, starte OCR...")
        
        texts = []
        for i, img in enumerate(images, 1):
            if i % 50 == 0:
                logger.info(f"    Seite {i}/{page_count}...")
            
            text = pytesseract.image_to_string(img, lang='deu+eng')
            texts.append(text)
        
        full_text = '\n'.join(texts)
        char_count = len(full_text)
        word_count = len(full_text.split())
        
        return full_text, char_count, word_count, page_count
        
    except Exception as e:
        logger.error(f"  ✗ Fehler bei OCR: {e}")
        return None, 0, 0, 0


def add_text_layer_with_ocrmypdf(pdf_path, txt_path, output_path):
    """Fügt Textebene mit ocrmypdf hinzu."""
    try:
        logger.info(f"  Füge Textebene hinzu...")
        
        result = ocrmypdf.ocr(
            input_file=str(pdf_path),
            output_file=str(output_path),
            language='deu+eng',
            force_ocr=True,
            optimize=0,
            keep_temporary_files=False,
            jobs=1,
            progress_bar=False,
            sidecar=str(txt_path)
        )
        
        if result != ocrmypdf.ExitCode.ok:
            logger.error(f"  ✗ ocrmypdf Fehler: {result}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Fehler beim Text-Layer: {e}")
        return False


# ============================================================================
# HAUPTVERARBEITUNG
# ============================================================================

def process_all_pdfs():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for pdf_path in INPUT_DIR.glob('*.pdf'): 
            future = executor.submit(process_pdf_with_tesseract, pdf_path)
            futures.append(future)
        results = [future.result() for future in futures]

    """Verarbeitet alle PDFs aus dem input Verzeichnis."""
    logger.info("\n" + "=" * 80)
    logger.info("2. OCR VERARBEITUNG")
    logger.info("=" * 80)
    
    # Finde alle PDFs
    pdf_files = sorted(INPUT_DIR.glob('*.pdf'))
    
    if not pdf_files:
        logger.warning(f"Keine PDF-Dateien in {INPUT_DIR} gefunden!")
        return {}
    
    logger.info(f"Gefunden: {len(pdf_files)} PDF-Dateien\n")
    
    statistics = {
        'total_files': len(pdf_files),
        'successful': 0,
        'failed': 0,
        'total_characters': 0,
        'total_words': 0,
        'total_pages': 0,
        'files': {}
    }
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{idx}/{len(pdf_files)}] Verarbeite: {pdf_path.name}")
        logger.info(f"  Größe: {pdf_path.stat().st_size / (1024*1024):.1f} MB")
        
        # OCR mit Tesseract
        text, char_count, word_count, page_count = process_pdf_with_tesseract(pdf_path)
        
        if text is None:
            logger.error(f"✗ Verarbeitung fehlgeschlagen")
            statistics['failed'] += 1
            continue
        
        # Speichern der txt-Datei
        base_name = pdf_path.stem
        txt_path = OUTPUT_DIR / f"{base_name}_tesseract.txt"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"  ✓ Text extrahiert: {char_count:,} Zeichen, {word_count:,} Wörter")
        
        # Textebene mit ocrmypdf hinzufügen
        output_pdf = OUTPUT_WITH_TEXT_DIR / f"{base_name}_with_text_layer.pdf"
        
        if add_text_layer_with_ocrmypdf(pdf_path, txt_path, output_pdf):
            logger.info(f"  ✓ Ausgabe gespeichert: {output_pdf.name}")
            
            statistics['successful'] += 1
            statistics['total_characters'] += char_count
            statistics['total_words'] += word_count
            statistics['total_pages'] += page_count
            
            statistics['files'][base_name] = {
                'input_file': pdf_path.name,
                'input_size_mb': round(pdf_path.stat().st_size / (1024*1024), 1),
                'pages': page_count,
                'characters': char_count,
                'words': word_count,
                'output_text': txt_path.name,
                'output_pdf': output_pdf.name,
                'output_size_mb': round(output_pdf.stat().st_size / (1024*1024), 1),
                'status': 'success'
            }
        else:
            logger.error(f"✗ Text-Layer Hinzufügung fehlgeschlagen")
            statistics['failed'] += 1
            statistics['files'][base_name] = {
                'status': 'failed',
                'error': 'Text layer creation failed'
            }
        
        logger.info("")
    
    return statistics


# ============================================================================
# STATISTIKEN & LOGGING
# ============================================================================

def save_statistics(statistics):
    """Speichert Statistiken in JSON und Log-Datei."""
    logger.info("=" * 80)
    logger.info("3. ABSCHLUSSBERICHT")
    logger.info("=" * 80)
    
    # JSON Statistiken
    stats_file = OUTPUT_DIR / 'ocr_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📊 GESAMTSTATISTIKEN:")
    logger.info(f"  Verarbeitete Dateien: {statistics['successful']}/{statistics['total_files']}")
    logger.info(f"  Fehler: {statistics['failed']}")
    logger.info(f"  Gesamte Seiten: {statistics['total_pages']:,}")
    logger.info(f"  Gesamte Zeichen: {statistics['total_characters']:,}")
    logger.info(f"  Gesamte Wörter: {statistics['total_words']:,}")
    
    logger.info(f"\n📁 DATEIEN:")
    logger.info(f"  Text-Ausgaben: {OUTPUT_DIR}")
    logger.info(f"  PDF mit Textebene: {OUTPUT_WITH_TEXT_DIR}")
    logger.info(f"  Statistiken: {stats_file}")
    logger.info(f"  Log-Datei: {log_file}")
    
    # Detaillierte Tabelle
    logger.info(f"\n📋 DETAILLIERTE ÜBERSICHT:")
    logger.info("-" * 100)
    logger.info(f"{'Datei':<25} {'Seiten':>8} {'Zeichen':>12} {'Wörter':>10} {'Input MB':>10} {'Output MB':>10}")
    logger.info("-" * 100)
    
    for name, info in statistics['files'].items():
        if info.get('status') == 'success':
            logger.info(
                f"{name:<25} {info['pages']:>8} {info['characters']:>12,} "
                f"{info['words']:>10,} {info['input_size_mb']:>10.1f} {info['output_size_mb']:>10.1f}"
            )
    
    logger.info("-" * 100)
    logger.info(f"{'GESAMT':<25} {statistics['total_pages']:>8} {statistics['total_characters']:>12,} "
                f"{statistics['total_words']:>10,}")
    logger.info("-" * 100)
    
    logger.info("\n✓ Pipeline erfolgreich abgeschlossen!")
    logger.info("=" * 80)


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def main():
    """Hauptfunktion."""
    try:
        # Installation
        if not install_requirements():
            logger.warning("Requirements Installation hatte Probleme, fahre fort...")
        
        # Verarbeitung
        statistics = process_all_pdfs()
        
        # Bericht
        save_statistics(statistics)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠ Pipeline abgebrochen durch Benutzer")
        return 1
        
    except Exception as e:
        logger.error(f"\n\n✗ Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
