# Basisimage
FROM python:3.12-slim

# Arbeitsverzeichnis
WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    ghostscript \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python-Dependencies kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Projekt kopieren
COPY . .

# Default Command
CMD ["python", "ocr_process.py"]
