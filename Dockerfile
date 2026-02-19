FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    ghostscript \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Wichtig: distro packages entfernen
RUN pip uninstall -y ocrmypdf pdfminer-six pikepdf || true

# pip aktualisieren
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Requirements installieren (erzwingt aktuelle Version)
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

CMD ["python", "ocr_process.py"]
