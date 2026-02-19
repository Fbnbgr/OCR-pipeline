import ocrmypdf
import concurrent.futures
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def process_pdf(pdf_path: Path):
    logger = get_logger(pdf_path.name)
    output_pdf = OUTPUT_DIR / f"{pdf_path.stem}_ocr.pdf"

    logger.info(f"Starte Verarbeitung → {output_pdf.name}")

    # OCRmyPDF nutzt intern das Python-logging-Modul.
    # Wir hängen unseren Handler an den ocrmypdf-Logger,
    # damit dessen Output unter dem Dateinamen erscheint.
    ocr_logger = logging.getLogger("ocrmypdf")
    ocr_logger.handlers.clear()
    ocr_logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt=f"%(asctime)s [{pdf_path.name}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    ocr_logger.addHandler(handler)
    ocr_logger.setLevel(logging.DEBUG)

    try:
        ocrmypdf.ocr(
            str(pdf_path),
            str(output_pdf),
            deskew=True,
            language="deu+chi_sim+chi_tra",
            optimize=1,
            jobs=1,
        )
        logger.info(f"Fertig ✓")
    except ocrmypdf.exceptions.PriorOcrFoundError:
        logger.warning("PDF enthält bereits OCR-Text – wird übersprungen.")
    except ocrmypdf.exceptions.EncryptedPdfError:
        logger.error("PDF ist verschlüsselt – kann nicht verarbeitet werden.")
    except Exception as e:
        logger.error(f"Fehler: {e}", exc_info=True)
    finally:
        ocr_logger.handlers.clear()


def process_all():
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.getLogger("main").warning(f"Keine PDFs in {INPUT_DIR} gefunden.")
        return

    root_logger = get_logger("main")
    root_logger.info(f"{len(pdf_files)} PDF(s) gefunden, starte Verarbeitung...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_pdf, pdf_files)

    root_logger.info("Alle Dateien verarbeitet.")


if __name__ == "__main__":
    process_all()